from collections import defaultdict
from datetime import datetime

import h5py
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pint
import znh5md
import zntrack
from ase.data import chemical_symbols
from jax import jit, vmap
from jax.numpy.fft import irfft, rfft
from tqdm import tqdm

ureg = pint.UnitRegistry()


@jit
def unwrap_positions(
    pos: jnp.ndarray, cells: jnp.ndarray, inv_cells: jnp.ndarray
) -> jnp.ndarray:
    frac = jnp.einsum("nij,nj->ni", inv_cells, pos)
    delta_frac = jnp.diff(frac, axis=0)
    delta_frac -= jnp.round(delta_frac)
    frac_unwrapped = jnp.concatenate(
        [frac[:1], frac[:1] + jnp.cumsum(delta_frac, axis=0)], axis=0
    )
    return jnp.einsum("nij,nj->ni", cells, frac_unwrapped)


@jit
def compute_msd_fft(
    x: jnp.ndarray, cell: jnp.ndarray, inv_cell: jnp.ndarray
) -> jnp.ndarray:
    x = unwrap_positions(x, cell, inv_cell)
    N = x.shape[0]

    norm2 = jnp.sum(x**2, axis=1)

    x_padded = jnp.concatenate([x, jnp.zeros_like(x)], axis=0)
    fx = rfft(x_padded, axis=0)
    acf = irfft(fx * jnp.conj(fx), axis=0)[:N]
    acf = jnp.sum(acf, axis=1)

    count = jnp.arange(N, 0, -1)
    msd = norm2[None, :] + norm2[:, None] - 2 * acf[:, None] / count[:, None]
    return jnp.mean(msd, axis=1)


class EinsteinSelfDiffusion(zntrack.Node):
    file: str = zntrack.deps_path()
    sampling_rate: int = zntrack.params()
    timestep: float = zntrack.params()
    batch_size: int = zntrack.params(64)

    def get_cells(self):
        io = znh5md.IO(
            self.file, variable_shape=False, include=["position", "box"], mask=[]
        )
        cells = [atoms.cell[:] for atoms in io[:]]
        cells = jnp.stack(cells)
        inv_cells = jnp.linalg.inv(cells)
        return cells, inv_cells

    def get_atomic_numbers(self) -> list[int]:
        io = znh5md.IO(self.file, variable_shape=False, include=["position"])
        return io[0].get_atomic_numbers().tolist()

    def get_positions(self, index: slice) -> jnp.ndarray:
        """Load unwrapped positions for a slice of atom indices."""
        print(f"{datetime.now()} Loading positions for index {index}")
        io = znh5md.IO(
            self.file, variable_shape=False, include=["position"], mask=index
        )
        pos = jnp.stack([atoms.positions for atoms in io[:]])
        print(f"{datetime.now()} Loaded positions shape: {pos.shape}")
        return pos  # shape: (n_frames, n_atoms_in_batch, 3)

    def compute_diffusion_coefficients(self, msds: dict, timestep_fs: float) -> dict:
        results = {}
        timestep_ps = timestep_fs / 1000  # fs -> ps

        for Z, msd_array in msds.items():
            msd_avg = jnp.mean(jnp.stack(msd_array), axis=0)
            time_ps = jnp.arange(msd_avg.shape[0]) * timestep_ps
            D = msd_avg / (6 * time_ps)

            # Handle division by zero in first frame
            D = jnp.where(time_ps > 0, D, 0)

            size = D.shape[0]
            start, end = int(size * 0.2), int(size * 0.8)
            D_avg = jnp.mean(D[start:end])

            results[Z] = {
                "diffusion_coefficient": D_avg,
                "msd": msd_avg,
                "time_ps": time_ps,
            }

            print(f"Z={Z}: D = {D_avg:.4f} Å²/ps")

        return results

    def plot_results(self, results: dict):
        n_species = len(results)
        n_cols = 3
        n_rows = (n_species + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]

        for ax, (Z, data) in zip(axes, results.items()):
            time_ps, msd = data["time_ps"], data["msd"]
            size = len(time_ps)
            start, end = int(size * 0.2), int(size * 0.8)

            fit_time = time_ps[start:end]
            fit_msd = msd[start:end]

            A = jnp.vstack([fit_time, jnp.ones_like(fit_time)]).T
            slope, intercept = jnp.linalg.lstsq(A, fit_msd, rcond=None)[0]
            fit_line = slope * time_ps + intercept
            D_fit = slope / 6

            ax.plot(time_ps, msd, label="MSD")
            ax.plot(
                time_ps,
                fit_line,
                "--",
                label=f"Fit: D = {D_fit:.4f} Å²/ps",
                color="tab:red",
            )
            ax.axvspan(
                time_ps[start],
                time_ps[end - 1],
                color="gray",
                alpha=0.2,
                label="Fit Window",
            )
            ax.set_title(f"MSD for Z={Z} ({chemical_symbols[int(Z)]})")
            ax.set_xlabel("Time [ps]")
            ax.set_ylabel("MSD [Å²]")
            ax.legend()
            ax.grid(True)

        for ax in axes[n_species:]:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig("msd_species.png")
        plt.show()

    def run(self):
        cells, inv_cells = self.get_cells()
        atomic_numbers = self.get_atomic_numbers()
        timestep_fs = self.timestep * ureg.fs * self.sampling_rate
        msds = defaultdict(list)

        n_atoms = len(atomic_numbers)

        for start in tqdm(range(0, n_atoms, self.batch_size)):
            end = min(start + self.batch_size, n_atoms)
            atom_slice = slice(start, end)
            Z_batch = atomic_numbers[start:end]

            pos = self.get_positions(atom_slice)  # shape: (n_frames, batch_size, 3)

            for local_idx, Z in enumerate(Z_batch):
                single_pos = pos[:, local_idx, :]  # shape: (n_frames, 3)
                msd = compute_msd_fft(single_pos, cells, inv_cells)
                msds[Z].append(msd)

        results = self.compute_diffusion_coefficients(msds, timestep_fs.magnitude)
        self.results = results
        self.plot_results(results)
