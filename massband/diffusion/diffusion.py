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

ureg = pint.UnitRegistry()
logger = logging.getLogger(__name__)

class EinsteinSelfDiffusion(zntrack.Node):
    """Compute self-diffusion coefficients using Einstein relation from MD trajectories."""

    file: str = zntrack.deps_path()
    sampling_rate: int = zntrack.params()
    timestep: float = zntrack.params()
    batch_size: int = zntrack.params(64)
    fit_window: Tuple[float, float] = zntrack.params((0.2, 0.8))
    method: Literal["direct", "fft"] = zntrack.params("fft")
    structures: list[str]|None = zntrack.params(None)

    def get_cells(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        io = znh5md.IO(
            self.file, variable_shape=False, include=["position", "box"], mask=[]
        )
        cells = jnp.stack([atoms.cell[:] for atoms in io])
        inv_cells = jnp.linalg.inv(cells)
        return cells, inv_cells

    def get_atomic_numbers(self) -> List[int]:
        """Get atomic numbers from the trajectory."""
        io = znh5md.IO(self.file, variable_shape=False, include=["position"])
        return io[0].get_atomic_numbers().tolist()

    def get_positions(self, index: slice) -> jnp.ndarray:
        """Load unwrapped positions for a slice of atom indices."""
        # TODO: for COM diffusion, we should process this here ?

        logger.info(f"Loading positions for index {index}")
        io = znh5md.IO(
            self.file, variable_shape=False, include=["position"], mask=index
        )
        pos = jnp.stack([atoms.positions for atoms in io])
        logger.info(f"Loaded positions shape: {pos.shape}")
        return pos  # shape: (n_frames, n_atoms_in_batch, 3)

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

            results[Z] = {
                "Z": Z,
                "symbol": chemical_symbols[Z],
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

            logger.info(f"Z={Z} ({chemical_symbols[Z]}): D = {D_fit:.4f} Å²/ps")

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
        cells, inv_cells = self.get_cells()

        if self.structures is not None:
            io = znh5md.IO(self.file, variable_shape=False)
            atoms = io[0]
            molecules = defaultdict(list)
            for structure in self.structures:
                indices = rdkit2ase.match_substructure(
                    atoms, structure
                )
                if len(indices) > 0:
                    molecules[structure].extend(indices)

            print(f"Found {molecules} structures in the trajectory.")
            # TODO: in this case, we don't want to compute the of each atom, but rather the center of mass of the molecule, given by the indices
            # - get the mass of each atom in the molecule from the initial structure using ase
            # - assume the indices are the same for all frames, iterate the dataset in the given batch size, for all indices not in the molecule, compute the MSD as usual
            # for all the others, store the positions in a dict[index] = positions, and as soon as we have all the positions for any molecule, compute the MSD for that molecule
            # using the COM and then remove the indices / positions from the dataset



        atomic_numbers = self.get_atomic_numbers()
        timestep_fs = (self.timestep * ureg.fs * self.sampling_rate).magnitude
        msds = defaultdict(list)

        n_atoms = len(atomic_numbers)
        logger.info(f"Starting MSD calculation for {n_atoms} atoms")

        if self.method == "direct":

            @jit
            def msd_fn(x, cell, inv_cell):
                x_unwrapped = unwrap_positions(x, cell, inv_cell)
                return compute_msd_direct(x_unwrapped)

            logger.info("Using direct MSD computation method")
        elif self.method == "fft":

            def msd_fn(x, cell, inv_cell):
                x_unwrapped = unwrap_positions(x, cell, inv_cell)
                return compute_msd_fft(x_unwrapped)

            logger.info("Using FFT-based MSD computation method")
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'direct' or 'fft'.")

        # Process atoms in batches
        for start in tqdm(range(0, n_atoms, self.batch_size), desc="Processing atoms"):
            end = min(start + self.batch_size, n_atoms)
            atom_slice = slice(start, end)
            Z_batch = atomic_numbers[start:end]

            pos = self.get_positions(atom_slice)  # shape: (n_frames, batch_size, 3)
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
