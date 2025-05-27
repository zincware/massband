import jax.numpy as jnp
import matplotlib.pyplot as plt
import pint
import znh5md
import zntrack
from ase.data import chemical_symbols
from jax import jit, vmap
from jax.numpy.fft import irfft, rfft

ureg = pint.UnitRegistry()


def compute_msd_fft(x: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the mean squared displacement (MSD) using FFT for a single particle.

    Parameters:
    x: jnp.ndarray of shape (N, 3), unwrapped trajectory for one particle

    Returns:
    msd: jnp.ndarray of shape (N,)
    """
    N = x.shape[0]

    norm2 = jnp.sum(x**2, axis=1)  # (N,)

    x_padded = jnp.concatenate([x, jnp.zeros_like(x)], axis=0)  # (2N, 3)
    fx = rfft(x_padded, axis=0)
    acf = irfft(fx * jnp.conj(fx), axis=0)[:N]  # (N, 3)
    acf = jnp.sum(acf, axis=1)  # (N,)

    count = jnp.arange(N, 0, -1)  # (N,)

    msd = norm2[None, :] + norm2[:, None] - 2 * acf[:, None] / count[:, None]  # (N, N)
    return jnp.mean(msd, axis=1)  # (N,)


class EinsteinSelfDiffusion(zntrack.Node):
    file: str = zntrack.deps_path()
    sampling_rate: int = zntrack.params()
    timestep: float = zntrack.params()

    def run(self):
        io = znh5md.IO(self.file, variable_shape=False, include=["position", "box"])
        size = len(io)
        batch_indices = range(0, size, 10)
        atomic_numbers = io[0].get_atomic_numbers()
        jit_compute_msd_fft = jit(compute_msd_fft)

        # Map atomic number → indices
        atom_groups = {}
        for i, Z in enumerate(atomic_numbers):
            atom_groups.setdefault(Z, []).append(i)

        # Analyze each group
        results = {}
        for Z, indices in atom_groups.items():
            print(f"Analyzing atomic number {Z} with {len(indices)} atoms")
            pos, cells = self._extract_trajectory(io, batch_indices, indices)
            unwrapped = self._unwrap_positions(pos, cells)  # shape (N, M, 3)

            # Map over particles: shape (M, N) → then mean over axis 0
            per_particle_msd = vmap(jit_compute_msd_fft, in_axes=1)(unwrapped)
            msd = jnp.mean(per_particle_msd, axis=0)  # shape (N,)

            timestep = self.timestep * ureg.fs * self.sampling_rate
            time_ps = jnp.arange(msd.shape[0]) * timestep.magnitude / 1000
            D = msd / (6 * time_ps)
            # exclude first 20 and last 20 percent of data
            size = D.shape[0]
            start = int(size * 0.2)
            end = int(size * 0.8)
            D = D[start:end]
            # average over the remaining data
            D_avg = jnp.mean(D)

            results[Z] = {
                "diffusion_coefficient": D_avg,
                "msd": msd,
                "time_ps": time_ps,
            }
            print(f"Z={Z}: D = {D_avg:.4f} Å²/ps")

        self.results = results

        # Plot MSD curves with fit window and fit line
        n_species = len(results)
        n_cols = 3
        n_rows = (n_species + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]

        for ax, (Z, data) in zip(axes, results.items()):
            time_ps = data["time_ps"]
            msd = data["msd"]

            # Fit window
            size = len(time_ps)
            start = int(size * 0.2)
            end = int(size * 0.8)

            fit_time = time_ps[start:end]
            fit_msd = msd[start:end]

            # Linear fit (least squares)
            A = jnp.vstack([fit_time, jnp.ones_like(fit_time)]).T
            slope, intercept = jnp.linalg.lstsq(A, fit_msd, rcond=None)[0]
            fit_line = slope * time_ps + intercept
            D_fit = slope / 6  # since MSD = 6Dt

            # Plot
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

        # Hide unused axes
        for ax in axes[n_species:]:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig("msd_species.png")
        plt.show()

    def _extract_trajectory(self, io, batch_indices, selected_indices):
        pos = []
        cells = []
        io.mask = selected_indices
        frames = io[:]
        for atoms in frames:
            pos.append(atoms.positions)
            cells.append(atoms.cell[:])
        return jnp.stack(pos), jnp.stack(cells)

    def _unwrap_positions(self, pos, cells):
        inv_cells = jnp.linalg.inv(cells)
        frac = jnp.einsum("nij,nmj->nmi", inv_cells, pos)
        delta_frac = jnp.diff(frac, axis=0)
        delta_frac -= jnp.round(delta_frac)
        frac_unwrapped = jnp.concatenate(
            [frac[:1], frac[:1] + jnp.cumsum(delta_frac, axis=0)], axis=0
        )
        return jnp.einsum("nij,nmj->nmi", cells, frac_unwrapped)
