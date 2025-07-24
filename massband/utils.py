import jax.numpy as jnp
import numpy as np
from jax import jit


def _validate_unwrap_inputs(
    positions: jnp.ndarray, cells: jnp.ndarray, inv_cells: jnp.ndarray
) -> None:
    """Validate inputs for unwrap_positions function.

    Parameters
    ----------
    positions : jnp.ndarray
        Array of atomic positions.
    cells : jnp.ndarray
        Array of unit cell vectors.
    inv_cells : jnp.ndarray
        Array of inverse unit cell vectors.

    Raises
    ------
    TypeError
        If input arrays are not jax arrays.
    ValueError
        If array shapes are incompatible or arrays are empty.
    """
    # Type validation
    if not isinstance(positions, jnp.ndarray):
        raise TypeError("positions must be a jax numpy array")
    if not isinstance(cells, jnp.ndarray):
        raise TypeError("cells must be a jax numpy array")
    if not isinstance(inv_cells, jnp.ndarray):
        raise TypeError("inv_cells must be a jax numpy array")

    # Empty array validation
    if positions.size == 0:
        raise ValueError("positions array cannot be empty")
    if cells.size == 0:
        raise ValueError("cells array cannot be empty")
    if inv_cells.size == 0:
        raise ValueError("inv_cells array cannot be empty")

    # Shape validation
    if positions.ndim != 3:
        raise ValueError(
            f"positions must be 3D array (n_frames, n_atoms, 3), got shape {positions.shape}"
        )
    if cells.ndim != 3 or cells.shape[-2:] != (3, 3):
        raise ValueError(
            f"cells must have shape (n_frames, 3, 3), got shape {cells.shape}"
        )
    if inv_cells.ndim != 3 or inv_cells.shape[-2:] != (3, 3):
        raise ValueError(
            f"inv_cells must have shape (n_frames, 3, 3), got shape {inv_cells.shape}"
        )

    # Frame consistency validation
    n_frames = positions.shape[0]
    if cells.shape[0] != n_frames:
        raise ValueError(f"cells must have {n_frames} frames, got {cells.shape[0]}")
    if inv_cells.shape[0] != n_frames:
        raise ValueError(
            f"inv_cells must have {n_frames} frames, got {inv_cells.shape[0]}"
        )

    # Spatial dimensions validation
    if positions.shape[-1] != 3:
        raise ValueError(
            f"positions must have 3 spatial dimensions, got {positions.shape[-1]}"
        )


def unwrap_positions(
    positions: jnp.ndarray, cells: jnp.ndarray, inv_cells: jnp.ndarray
) -> jnp.ndarray:
    """Unwrap atomic positions to account for periodic boundary conditions.

    Parameters
    ----------
    positions : jnp.ndarray
        Array of atomic positions with shape (n_frames, n_atoms, 3).
    cells : jnp.ndarray
        Array of unit cell vectors with shape (n_frames, 3, 3).
    inv_cells : jnp.ndarray
        Array of inverse unit cell vectors with shape (n_frames, 3, 3).

    Returns
    -------
    jnp.ndarray
        Unwrapped positions with the same shape as input positions.

    Raises
    ------
    ValueError
        If array shapes are incompatible or arrays are empty.
    TypeError
        If input arrays are not jax arrays.

    Notes
    -----
    This implementation assumes orthogonal cells and is not general
    enough for all cell types (e.g., triclinic cells).
    """
    _validate_unwrap_inputs(positions, cells, inv_cells)

    # Calculate displacements
    displacements = jnp.diff(positions, axis=0)

    # Rewrap displacements into the simulation box
    scaled_displacements = jnp.einsum("...ij,...j->...i", inv_cells[:-1], displacements)
    rewrapped_displacements = scaled_displacements - jnp.round(scaled_displacements)
    unscaled_displacements = jnp.einsum(
        "...ij,...j->...i", cells[:-1], rewrapped_displacements
    )

    # Cumulatively sum the unwrapped displacements to get the unwrapped trajectory
    unwrapped_positions = jnp.concatenate(
        [
            positions[0][jnp.newaxis, ...],
            positions[0] + jnp.cumsum(unscaled_displacements, axis=0),
        ],
        axis=0,
    )
    return unwrapped_positions


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
