from collections import defaultdict
import logging
from typing import Dict, List, Tuple, Literal

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pint
import znh5md
import zntrack
from ase.data import chemical_symbols
from jax import jit, vmap
from tqdm import tqdm
from massband.utils import unwrap_positions
import rdkit2ase
from massband.diffusion.utils import compute_msd_direct, compute_msd_fft
import jax

# Enable 64-bit precision in JAX for FFT accuracy
jax.config.update("jax_enable_x64", True)

ureg = pint.UnitRegistry()
log = logging.getLogger(__name__)


class EinsteinSelfDiffusion(zntrack.Node):
    """Compute self-diffusion coefficients using Einstein relation from MD trajectories."""

    file: str = zntrack.deps_path()
    sampling_rate: int = zntrack.params()
    timestep: float = zntrack.params()
    batch_size: int = zntrack.params(64)
    fit_window: Tuple[float, float] = zntrack.params((0.2, 0.8))
    method: Literal["direct", "fft"] = zntrack.params("fft")
    structures: list[str] | None = zntrack.params(None)
    # TODO: allow not smiles but just the sum formula, e.g. "C6H12O6" for glucose

    def get_cells(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        io = znh5md.IO(
            self.file, variable_shape=False, include=["position", "box"], mask=[]
        )
        cells = jnp.stack([atoms.cell[:] for atoms in io[:]])
        inv_cells = jnp.linalg.inv(cells)
        return cells, inv_cells

    def get_atomic_numbers(self) -> List[int]:
        """Get atomic numbers from the trajectory."""
        io = znh5md.IO(self.file, variable_shape=False, include=["position"])
        return io[0].get_atomic_numbers().tolist()

    def get_masses(self) -> List[float]:
        """Get atomic masses from the trajectory."""
        io = znh5md.IO(self.file, variable_shape=False, include=["position"])
        return io[0].get_masses().tolist()

    def get_positions(self, index: slice) -> jnp.ndarray:
        """Load unwrapped positions for a slice of atom indices."""
        # TODO: for COM diffusion, we should process this here ?

        log.info(f"Loading positions for index {index}")
        io = znh5md.IO(
            self.file, variable_shape=False, include=["position"], mask=index
        )
        pos = jnp.stack([atoms.positions for atoms in io[:]])
        log.info(f"Loaded positions shape: {pos.shape}")
        return pos  # shape: (n_frames, n_atoms_in_batch, 3)

    def postprocess_positions(
        self,
        pos: jnp.ndarray,
        masses,
        atomic_numbers,
        substructures,
        atom_slice: slice,
        max_cache_size: int = 10000,
    ):
        # pos: (n_frames, n_atoms_in_batch, 3)
        # masses: (n_total_atoms,)
        # atomic_numbers: (n_total_atoms,)
        # substructures: dict[substructure_name -> list[tuple[atom_indices]]]
        # atom_slice: slice (start, stop) for current batch of atoms

        values = []
        Zs = []

        try:
            cache = getattr(self, "data_cache")
        except AttributeError:
            cache = {}

        # Track all molecular atom indices
        molecular_indices = set()
        mol_index_map = {}  # atom idx -> list of (substructure_name, mol_indices)

        for sub_name, mols in substructures.items():
            for mol_indices in mols:
                for idx in mol_indices:
                    mol_index_map.setdefault(idx, []).append(
                        (sub_name, tuple(mol_indices))
                    )
                    molecular_indices.add(idx)

        # Cache positions of molecular atoms, append standalone atoms directly
        for i in range(atom_slice.start, atom_slice.stop):
            rel_i = i - atom_slice.start
            pos_i = pos[:, rel_i, :]  # (n_frames, 3)

            if i in molecular_indices:
                cache[i] = pos_i
            else:
                Z = atomic_numbers[i]
                values.append(pos_i)
                Zs.append(Z)

        # Enforce cache size
        if len(cache) > max_cache_size:
            raise RuntimeError(f"Cache size exceeded max limit of {max_cache_size}")

        # Track used molecules so we can remove them from cache
        used_mol_keys = set()

        for sub_name, mols in substructures.items():
            for mol_indices in mols:
                if all(idx in cache for idx in mol_indices):
                    # All atoms in molecule are available, compute COM
                    pos_stack = jnp.stack(
                        [cache[idx] * masses[idx] for idx in mol_indices], axis=0
                    )  # (n_atoms, n_frames, 3)
                    total_mass = jnp.sum(
                        jnp.array([masses[idx] for idx in mol_indices])
                    )
                    com = jnp.sum(pos_stack, axis=0) / total_mass  # (n_frames, 3)

                    values.append(com)
                    Zs.append(sub_name)

                    used_mol_keys.add(tuple(mol_indices))

        # Clear used entries from cache
        for mol_indices in used_mol_keys:
            for idx in mol_indices:
                cache.pop(idx, None)
        print(
            f"Used {len(used_mol_keys)} molecular indices, remaining cache size: {len(cache)}"
        )

        self.data_cache = cache

        values = jnp.stack(values, axis=1)  # shape: (n_frames, n_entities, 3)
        return Zs, values

    def compute_diffusion_coefficients(
        self, msds: Dict[int, List[jnp.ndarray]], timestep_fs: float
    ) -> Dict:
        """Calculate diffusion coefficients from MSD data.

        Args:
            msds: Dictionary mapping atomic numbers to lists of MSD arrays.
            timestep_fs: MD timestep in femtoseconds.

        Returns:
            Dictionary containing full data for plotting and analysis.
        """
        results = {}
        timestep_ps = timestep_fs / 1000  # fs -> ps

        for Z, msd_array in msds.items():
            msd_avg = jnp.mean(jnp.stack(msd_array), axis=0)
            time_ps = jnp.arange(msd_avg.shape[0]) * timestep_ps

            # Fit to linear region
            start_idx = int(len(time_ps) * self.fit_window[0])
            end_idx = int(len(time_ps) * self.fit_window[1])

            fit_time = time_ps[start_idx:end_idx]
            fit_msd = msd_avg[start_idx:end_idx]

            # Linear least squares fit
            A = jnp.vstack([fit_time, jnp.ones_like(fit_time)]).T
            slope, intercept = jnp.linalg.lstsq(A, fit_msd, rcond=None)[0]
            D_fit = slope / 6
            fit_line = slope * time_ps + intercept
            try:
                symbol = chemical_symbols[Z]
            except TypeError:
                symbol = Z

            results[Z] = {
                "Z": Z,
                "symbol": symbol,
                "diffusion_coefficient": D_fit,
                "time_ps": time_ps,
                "msd": msd_avg,
                "fit_line": fit_line,
                "fit_window": (start_idx, end_idx),
                "fit_msd": fit_msd,
                "fit_time": fit_time,
                "slope": slope,
                "intercept": intercept,
            }

            log.info(f"Z={Z} ({symbol}): D = {D_fit:.4f} Å²/ps")

        return results

    def plot_results(self, results: Dict, filename: str = "msd_species.png"):
        """Plot MSD curves and diffusion coefficient fits.

        Args:
            results: Dictionary containing precomputed MSD and fit data
            filename: Output filename for the plot
        """
        n_species = len(results)
        n_cols = min(3, n_species)
        n_rows = (n_species + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for ax, (Z, data) in zip(axes, results.items()):
            time_ps = data["time_ps"]
            msd = data["msd"]
            fit_line = data["fit_line"]
            start_idx, end_idx = data["fit_window"]

            ax.plot(time_ps, msd, label="MSD")
            ax.plot(
                time_ps,
                fit_line,
                "--",
                label=f"Fit: D = {data['diffusion_coefficient']:.4f} Å²/ps",
                color="tab:red",
            )
            ax.axvspan(
                time_ps[start_idx],
                time_ps[end_idx - 1],
                color="gray",
                alpha=0.2,
                label="Fit Window",
            )

            ax.set_title(f"{data['symbol']} (Z={Z})")
            ax.set_xlabel("Time (ps)")
            ax.set_ylabel("MSD (Å²)")
            ax.legend()
            ax.grid(True)

        # Hide unused subplots
        for ax in axes[n_species:]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

    def run(self):
        """Main computation workflow."""
        log.info("Collecting cell vectors and inverse cell vectors")
        cells, inv_cells = self.get_cells()

        substructures = defaultdict(list)
        # dict[str, list[tuple[int, ...]]]
        if self.structures is not None:
            io = znh5md.IO(self.file, variable_shape=False)
            atoms = io[0]
            log.info(f"Searching for substructures in {len(self.structures)} patterns")
            for structure in self.structures:
                indices = rdkit2ase.match_substructure(
                    atoms,
                    smiles=structure,
                    suggestions=self.structures,
                )
                if len(indices) > 0:
                    substructures[structure].extend(indices)

                log.info(
                    f"Found {len(indices)} matches for substructure {structure} in the dataset."
                )
            # TODO: in this case, we don't want to compute the of each atom, but rather the center of mass of the molecule, given by the indices
            # - get the mass of each atom in the molecule from the initial structure using ase
            # - assume the indices are the same for all frames, iterate the dataset in the given batch size, for all indices not in the molecule, compute the MSD as usual
            # for all the others, store the positions in a dict[index] = positions, and as soon as we have all the positions for any molecule, compute the MSD for that molecule
            # using the COM and then remove the indices / positions from the dataset
            # have a max size of the data cache and raise an error, also allow direct indexing as an alternative.

        log.info("Collecting atomic masses and numbers")
        masses = self.get_masses()
        atomic_numbers = self.get_atomic_numbers()
        # use iddentifier and not aotmic numbers so one can also use com
        timestep_fs = (self.timestep * ureg.fs * self.sampling_rate).magnitude
        msds = defaultdict(list)

        n_atoms = len(atomic_numbers)
        log.info(f"Starting MSD calculation for {n_atoms} atoms")

        if self.method == "direct":

            @jit
            def msd_fn(x, cell, inv_cell):
                # x_unwrapped = unwrap_positions(x, cell, inv_cell)
                return compute_msd_direct(x)

            log.info("Using direct MSD computation method")
        elif self.method == "fft":

            def msd_fn(x, cell, inv_cell):
                # x_unwrapped = unwrap_positions(x, cell, inv_cell)
                return compute_msd_fft(x)

            log.info("Using FFT-based MSD computation method")
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'direct' or 'fft'.")

        # Process atoms in batches
        for start in tqdm(range(0, n_atoms, self.batch_size), desc="Processing atoms"):
            end = min(start + self.batch_size, n_atoms)
            atom_slice = slice(start, end)
            Z_batch = atomic_numbers[start:end]

            pos = self.get_positions(atom_slice)  # shape: (n_frames, batch_size, 3)
            # postprocess positions for substructures
            # TODO: need to unwrap here!!
            # pos = unwrap_positions(pos, cells, inv_cells)
            # TODO: fix unnecessary multiple transposes
            pos = jnp.transpose(pos, (1, 0, 2))
            pos = vmap(lambda x: unwrap_positions(x, cells, inv_cells))(pos)
            pos = jnp.transpose(pos, (1, 0, 2))
            Z_batch, pos = self.postprocess_positions(
                pos, masses, atomic_numbers, substructures, atom_slice
            )
            # each step print the shape of the cached pos
            if self.method == "direct":
                pos = jnp.transpose(pos, (1, 0, 2))
                # TODO, specify vmap axis instead of transposing
                results = vmap(lambda x: msd_fn(x, cells, inv_cells))(pos)
            else:
                # TODO: vmap this stuff
                results = []
                for atom_index in range(pos.shape[1]):
                    pos_i = pos[:, atom_index, :]
                    msd_i = msd_fn(pos_i, cells, inv_cells)
                    results.append(msd_i)
            for msd, Z in zip(results, Z_batch):
                msds[Z].append(msd)

        results = self.compute_diffusion_coefficients(msds, timestep_fs)
        self.results = results
        self.plot_results(results)
