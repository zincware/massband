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
import jax.lax as jlax
from tqdm import tqdm
from massband.utils import unwrap_positions

ureg = pint.UnitRegistry()
logger = logging.getLogger(__name__)


@jit
def compute_msd_direct(
    x: jnp.ndarray, cell: jnp.ndarray, inv_cell: jnp.ndarray
) -> jnp.ndarray:
    """Compute the Mean Squared Displacement (MSD) using direct method.
    
    Parameters
    ----------
    x : jnp.ndarray
        Positions of a single atom in cartesian coordinates, shape (n_frames, 3).
    cell : jnp.ndarray
        Cell vectors for each frame, shape (n_frames, 3, 3).
    inv_cell : jnp.ndarray
        Inverse cell vectors for each frame, shape (n_frames, 3, 3).

    Returns
    -------
    jnp.ndarray
        Mean Squared Displacement for each time step, shape (n_frames,).
    """
    x_unwrapped = unwrap_positions(x, cell, inv_cell)
    N = x_unwrapped.shape[0]

    def msd_at_dt(dt):
        displacements = x_unwrapped[dt:] - x_unwrapped[:-dt]
        squared_displacements = jnp.sum(displacements**2, axis=1)
        return jnp.mean(squared_displacements)

    msd = jnp.zeros(N)

    # Avoid looping over dt=0 because displacement is zero
    msd = msd.at[0].set(0.0)
    msd_values = jnp.array([msd_at_dt(dt) for dt in range(1, N)])
    msd = msd.at[1:].set(msd_values)

    return msd


class EinsteinSelfDiffusion(zntrack.Node):
    """Compute self-diffusion coefficients using Einstein relation from MD trajectories."""

    file: str = zntrack.deps_path()
    sampling_rate: int = zntrack.params()
    timestep: float = zntrack.params()
    batch_size: int = zntrack.params(64)
    fit_window: Tuple[float, float] = zntrack.params((0.2, 0.8))
    method: Literal["direct"] = zntrack.params("direct")

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
            msds: Dictionary mapping atomic numbers to MSD arrays
            timestep_fs: MD timestep in femtoseconds

        Returns:
            Dictionary containing diffusion coefficients and related data
        """
        results = {}
        timestep_ps = timestep_fs / 1000  # fs -> ps

        for Z, msd_array in msds.items():
            # TODO: self.fit_window is not used here
            # TODO: save fit parameters for plotting later
            msd_avg = jnp.mean(jnp.stack(msd_array), axis=0)
            time_ps = jnp.arange(msd_avg.shape[0]) * timestep_ps
            D = jnp.where(time_ps > 0, msd_avg / (6 * time_ps), 0)

            # Fit to linear region
            start_idx = int(len(time_ps) * self.fit_window[0])
            end_idx = int(len(time_ps) * self.fit_window[1])
            D_avg = jnp.mean(D[start_idx:end_idx])

            results[Z] = {
                "diffusion_coefficient": D_avg,
                "msd": msd_avg,
                "time_ps": time_ps,
                "fit_window": (start_idx, end_idx),
            }

            logger.info(f"Z={Z} ({chemical_symbols[Z]}): D = {D_avg:.4f} Å²/ps")

        return results

    def plot_results(self, results: Dict, filename: str = "msd_species.png"):
        """Plot MSD curves and diffusion coefficient fits.

        Args:
            results: Dictionary of results from compute_diffusion_coefficients
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
            time_ps, msd = data["time_ps"], data["msd"]
            start_idx, end_idx = data["fit_window"]

            # Linear fit
            fit_time = time_ps[start_idx:end_idx]
            fit_msd = msd[start_idx:end_idx]

            A = jnp.vstack([fit_time, jnp.ones_like(fit_time)]).T
            slope, intercept = jnp.linalg.lstsq(A, fit_msd, rcond=None)[0]
            fit_line = slope * time_ps + intercept
            D_fit = slope / 6

            # Plotting
            ax.plot(time_ps, msd, label="MSD")
            ax.plot(
                time_ps,
                fit_line,
                "--",
                label=f"Fit: D = {D_fit:.4f} Å²/ps",
                color="tab:red",
            )
            ax.axvspan(
                time_ps[start_idx],
                time_ps[end_idx - 1],
                color="gray",
                alpha=0.2,
                label="Fit Window",
            )

            ax.set_title(f"{chemical_symbols[Z]} (Z={Z})")
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
        atomic_numbers = self.get_atomic_numbers()
        timestep_fs = (self.timestep * ureg.fs * self.sampling_rate).magnitude
        msds = defaultdict(list)

        n_atoms = len(atomic_numbers)
        logger.info(f"Starting MSD calculation for {n_atoms} atoms")

        # Process atoms in batches
        for start in tqdm(range(0, n_atoms, self.batch_size), desc="Processing atoms"):
            end = min(start + self.batch_size, n_atoms)
            atom_slice = slice(start, end)
            Z_batch = atomic_numbers[start:end]

            pos = self.get_positions(atom_slice)  # shape: (n_frames, batch_size, 3)
            pos = jnp.transpose(pos, (1, 0, 2))
            results = vmap(lambda x: compute_msd_direct(x, cells, inv_cells))(pos)
            for msd, Z in zip(results, Z_batch):
                msds[Z].append(msd)

        results = self.compute_diffusion_coefficients(msds, timestep_fs)
        self.results = results
        self.plot_results(results)
