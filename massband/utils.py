import jax.numpy as jnp
from jax import jit


@jit
def unwrap_positions(
    pos: jnp.ndarray, cells: jnp.ndarray, inv_cells: jnp.ndarray
) -> jnp.ndarray:
    """Unwrap positions using the inverse cell matrix.

    Parameters
    ----------
    pos : jnp.ndarray
        Positions of atoms in cartesian coordinates, shape (n_frames, 3).
    cells : jnp.ndarray
        Cell vectors for each frame, shape (n_frames, 3, 3).
    inv_cells : jnp.ndarray
        Inverse cell vectors for each frame, shape (n_frames, 3, 3).

    Returns
    -------
    jnp.ndarray
        Unwrapped positions in cartesian coordinates, shape (n_frames, 3).
    """
    # Convert to fractional coordinates per frame
    frac = jnp.einsum("nij,nj->ni", inv_cells, pos)

    # Compute wrapped displacements
    delta_frac = jnp.diff(frac, axis=0)
    delta_frac -= jnp.round(delta_frac)

    # Reconstruct unwrapped fractional coords
    frac_unwrapped = jnp.concatenate(
        [frac[:1], frac[:1] + jnp.cumsum(delta_frac, axis=0)], axis=0
    )

    # Back to cartesian coordinates
    pos_unwrapped = jnp.einsum("nij,nj->ni", cells, frac_unwrapped)
    return pos_unwrapped


@jit
def wrap_positions(pos: jnp.ndarray, cells: jnp.ndarray) -> jnp.ndarray:
    """Wrap positions into the unit cell.

    Parameters
    ----------
    pos : jnp.ndarray
        Shape (n_frames, n_atoms, 3)
    cells : jnp.ndarray
        Shape (n_frames, 3, 3)

    Returns
    -------
    jnp.ndarray
        Wrapped positions in Cartesian coordinates, shape (n_frames, n_atoms, 3)
    """
    # raise ValueError(pos.shape, cells.shape)
    # Convert to fractional: frac = pos @ inv(cell)
    inv_cells = jnp.linalg.inv(cells)
    frac = jnp.einsum("fij,faj->fai", inv_cells, pos)  # (n_frames, n_atoms, 3)

    # Wrap fractional positions
    frac_wrapped = frac % 1.0

    # Convert back to cartesian: pos = frac @ cell
    pos_wrapped = jnp.einsum("fij,faj->fai", cells, frac_wrapped)
    return pos_wrapped
