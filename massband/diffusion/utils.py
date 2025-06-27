import jax.numpy as jnp
from jax import jit
import numpy as np


@jit
def compute_msd_direct(x: jnp.ndarray) -> jnp.ndarray:
    """Compute the Mean Squared Displacement (MSD) using direct method.

    Parameters
    ----------
    x : jnp.ndarray
        Positions of a single atom in cartesian coordinates, shape (n_frames, 3).
    Returns
    -------
    jnp.ndarray
        Mean Squared Displacement for each time step, shape (n_frames,).
    """

    N = x.shape[0]

    def msd_at_dt(dt):
        displacements = x[dt:] - x[:-dt]
        squared_displacements = jnp.sum(displacements**2, axis=1)
        return jnp.mean(squared_displacements)

    msd = jnp.zeros(N)

    # Avoid looping over dt=0 because displacement is zero
    msd = msd.at[0].set(0.0)
    msd_values = jnp.array([msd_at_dt(dt) for dt in range(1, N)])
    msd = msd.at[1:].set(msd_values)

    return msd


def autocorrelation_1d_jax(data, N, n_fft):
    # Pad the signal with zeros
    padded_data = jnp.zeros(2 * n_fft).at[:N].set(data)

    # FFT → multiply by conjugate → IFFT → normalize
    fft_data = jnp.fft.fft(padded_data)
    autocorr = jnp.fft.ifft(fft_data * jnp.conj(fft_data))[:N].real
    norm = N - jnp.arange(N)  # Normalization factor
    return autocorr / norm


autocorrelation_1d_jax_jit = jit(autocorrelation_1d_jax, static_argnames=("N", "n_fft"))


def fn1(rsq, SAB, SUMSQ, N):
    MSD_0 = SUMSQ - 2 * SAB[0] * N

    cs1 = jnp.cumsum(rsq)[:-1]

    rsq_tail_reversed = rsq[1:][::-1]
    cs2 = jnp.cumsum(rsq_tail_reversed)

    denom = N - 1 - jnp.arange(N - 1)
    MSD_rest = (SUMSQ - cs1 - cs2) / denom
    MSD_rest -= 2 * SAB[1:]

    MSD = jnp.concatenate([jnp.array([MSD_0]), MSD_rest])
    return MSD


def _compute_msd_fft(pos: np.ndarray, n_fft: int, N: int) -> jnp.ndarray:
    rsq = jnp.sum(pos**2, axis=1)

    SAB = autocorrelation_1d_jax(pos[:, 0], N, n_fft)
    for i in range(1, pos.shape[1]):
        SAB += autocorrelation_1d_jax(pos[:, i], N, n_fft)

    SUMSQ = 2 * np.sum(rsq)

    MSD = fn1(rsq, SAB, SUMSQ, N)

    return MSD


def compute_msd_fft(positions: np.ndarray) -> np.ndarray:
    pos = np.asarray(positions)
    N = len(pos)
    n_fft = 2 ** (jnp.ceil(jnp.log2(N))).astype(int).item()
    if pos.shape[0] == 0:
        return np.array([], dtype=pos.dtype)
    if pos.ndim == 1:
        pos = pos.reshape((-1, 1))

    _compute_msd_fft_jit = jit(_compute_msd_fft, static_argnames=("n_fft", "N"))

    return _compute_msd_fft_jit(pos.reshape((N, -1)), n_fft=n_fft, N=N)
