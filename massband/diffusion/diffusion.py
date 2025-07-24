import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pint
import znh5md
import zntrack
from ase.data import chemical_symbols
from jax import jit
from tqdm import tqdm

from massband.com import center_of_mass_trajectories
from massband.utils import compute_msd_direct, compute_msd_fft, unwrap_positions

# Enable 64-bit precision in JAX for FFT accuracy
jax.config.update("jax_enable_x64", True)

ureg = pint.UnitRegistry()
log = logging.getLogger(__name__)


class EinsteinSelfDiffusion(zntrack.Node):
    """Compute self-diffusion coefficients using Einstein relation from MD trajectories."""

    file: str | Path = zntrack.deps_path()
    sampling_rate: int = zntrack.params()
    timestep: float = zntrack.params()
    batch_size: int = zntrack.params(64)
    fit_window: Tuple[float, float] = zntrack.params((0.2, 0.8))
    method: Literal["direct", "fft"] = zntrack.params("fft")
    structures: list[str] | None = zntrack.params(None)
    use_com: bool = zntrack.params(False)

    def get_cells(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        io = znh5md.IO(
            self.file, variable_shape=False, include=["position", "box"], mask=[]
        )
        cells = jnp.stack([atoms.cell[:] for atoms in io[:]])
        inv_cells = jnp.linalg.inv(cells)
        return cells, inv_cells

    def get_positions(self) -> jnp.ndarray:
        """Load unwrapped positions."""
        log.info("Loading all positions")
        io = znh5md.IO(self.file, variable_shape=False, include=["position"])
        pos = jnp.stack([atoms.positions for atoms in io[:]])
        log.info(f"Loaded positions shape: {pos.shape}")
        return pos  # shape: (n_frames, n_atoms, 3)

    def compute_diffusion_coefficients(
        self, msds: Dict[str | int, List[jnp.ndarray]], timestep_fs: float
    ) -> Dict:
        """Calculate diffusion coefficients from MSD data."""
        results = {}
        timestep_ps = timestep_fs / 1000  # fs -> ps

        for Z, msd_array in msds.items():
            msd_avg = jnp.mean(jnp.stack(msd_array), axis=0)
            time_ps = jnp.arange(msd_avg.shape[0]) * timestep_ps

            start_idx = int(len(time_ps) * self.fit_window[0])
            end_idx = int(len(time_ps) * self.fit_window[1])

            fit_time = time_ps[start_idx:end_idx]
            fit_msd = msd_avg[start_idx:end_idx]

            A = jnp.vstack([fit_time, jnp.ones_like(fit_time)]).T
            slope, intercept = jnp.linalg.lstsq(A, fit_msd, rcond=None)[0]
            D_fit = slope / 6
            fit_line = slope * time_ps + intercept
            symbol = chemical_symbols[Z] if isinstance(Z, int) else Z

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
        """Plot MSD curves and diffusion coefficient fits."""
        n_species = len(results)
        n_cols = min(3, n_species)
        n_rows = (n_species + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = [axes] if n_rows == 1 and n_cols == 1 else axes.flatten()

        for ax, (Z, data) in zip(axes, results.items()):
            time_ps, msd, fit_line = data["time_ps"], data["msd"], data["fit_line"]
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

        for ax in axes[n_species:]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

    def run(self):  # noqa: C901
        """Main computation workflow."""
        log.info("Collecting cell vectors and inverse cell vectors")
        cells, inv_cells = self.get_cells()

        timestep_fs = (self.timestep * ureg.fs * self.sampling_rate).magnitude
        msds = defaultdict(list)

        if self.use_com or self.structures:
            log.info("Computing center of mass trajectories")
            entity_positions, _ = center_of_mass_trajectories(
                file=self.file, structures=self.structures, wrap=False
            )
            entities_to_process = [
                (identifier, positions_array[:, i, :])
                for identifier, positions_array in entity_positions.items()
                for i in range(positions_array.shape[1])
            ]
            log.info(f"Starting MSD calculation for {len(entities_to_process)} entities")

        else:
            log.info("Computing atomic trajectories")
            io = znh5md.IO(self.file, variable_shape=False, include=["position"])
            frames = io[:]
            atomic_numbers = jnp.array(frames[0].get_atomic_numbers())
            positions = unwrap_positions(
                jnp.stack([atoms.positions for atoms in frames]), cells, inv_cells
            )
            entities_to_process = [
                (atomic_numbers[i].item(), positions[:, i, :])
                for i in range(positions.shape[1])
            ]
            log.info(f"Starting MSD calculation for {len(entities_to_process)} atoms")

        msd_fn = {
            "direct": jit(compute_msd_direct),
            "fft": compute_msd_fft,
        }.get(self.method)

        if msd_fn is None:
            raise ValueError(f"Unknown method: {self.method}. Use 'direct' or 'fft'.")

        log.info(f"Using {self.method} MSD computation method")

        for identifier, pos_i in tqdm(entities_to_process, desc="Processing entities"):
            msd_i = msd_fn(pos_i)
            msds[identifier].append(msd_i)

        results = self.compute_diffusion_coefficients(msds, timestep_fs)
        self.results = results
        self.plot_results(results)
