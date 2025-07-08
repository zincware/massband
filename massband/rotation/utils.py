import jax
import jax.numpy as jnp


@jax.jit
def compute_com(mol_positions, mol_masses):
    weighted_positions = mol_positions * mol_masses[None, :, None]
    return jnp.sum(weighted_positions, axis=1) / jnp.sum(mol_masses)


@jax.jit
def compute_orientation(positions, mol_masses, ref_orientation=None):
    # Center positions
    com = jnp.sum(positions * mol_masses[:, None], axis=0) / jnp.sum(mol_masses)
    centered = positions - com[None, :]

    if len(mol_masses) == 2:  # Diatomic
        orient = centered[1] - centered[0]
    else:  # Polyatomic
        # Vectorized inertia tensor computation
        r = centered
        r_sq = jnp.sum(r**2, axis=1)
        inertia = jnp.sum(mol_masses * r_sq) * jnp.eye(3) - (
            mol_masses[:, None, None] * r[:, :, None] * r[:, None, :]
        ).sum(axis=0)
        _, eigenvectors = jnp.linalg.eigh(inertia)
        orient = eigenvectors[:, 0]

    # Normalize and align with reference if provided
    orient = orient / jnp.linalg.norm(orient)
    if ref_orientation is not None and jnp.dot(orient, ref_orientation) < 0:
        orient = -orient
    return orient


@jax.jit
def compute_all_orientations(mol_positions, mol_masses, ref_orientation):
    # Compute COMs for all frames
    coms = jnp.sum(mol_positions * mol_masses[None, :, None], axis=1) / jnp.sum(
        mol_masses
    )
    centered = mol_positions - coms[:, None, :]

    if len(mol_masses) == 2:
        orients = centered[:, 1] - centered[:, 0]
    else:
        # Vectorized inertia tensor computation for all frames
        r = centered
        r_sq = jnp.sum(r**2, axis=2)
        eye = jnp.eye(3)

        # Compute inertia tensors for all frames
        term1 = (mol_masses[None, :] * r_sq).sum(axis=1)[:, None, None] * eye[None, :, :]
        term2 = (
            mol_masses[None, :, None, None] * r[:, :, :, None] * r[:, :, None, :]
        ).sum(axis=1)
        inertia = term1 - term2

        # Compute eigenvectors for all inertia tensors
        _, eigenvectors = jnp.linalg.eigh(inertia)
        orients = eigenvectors[:, :, 0]

    # Normalize and align with reference
    norms = jnp.linalg.norm(orients, axis=1, keepdims=True)
    orients = orients / norms
    flip = jnp.dot(orients, ref_orientation) < 0
    orients = jnp.where(flip[:, None], -orients, orients)
    return orients


@jax.jit
def unwrap_angles(p, period=2 * jnp.pi):
    """Unwraps a sequence of angles to make them continuous."""

    # The first argument to scan is the carry, the second is the current value
    def scan_fn(carry, x):
        # carry = (previous_angle, cumulative_correction)
        # x = current_angle
        prev_p, offset = carry
        dp = x - prev_p
        # Identify jumps larger than half the period
        offset += jnp.where(dp > period / 2, -period, 0.0)
        offset += jnp.where(dp < -period / 2, period, 0.0)
        return (x, offset), x + offset

    # Initialize carry with the first angle and zero correction
    _, unwrapped = jax.lax.scan(scan_fn, (p[0], 0.0), p)
    return unwrapped


# JAX implementation of orientation_to_euler
@jax.jit
def _orientation_to_euler_single(orientation):
    """Converts a single orientation vector to Euler angles (ZYX convention)."""
    x_axis = orientation / jnp.linalg.norm(orientation)

    # Create an arbitrary perpendicular vector for the y-axis
    y_axis = jnp.where(
        jnp.abs(x_axis[0]) < 0.9,
        jnp.cross(x_axis, jnp.array([1.0, 0.0, 0.0])),
        jnp.cross(x_axis, jnp.array([0.0, 1.0, 0.0])),
    )
    y_axis /= jnp.linalg.norm(y_axis)

    # The z-axis completes the orthonormal basis
    z_axis = jnp.cross(x_axis, y_axis)

    # Construct the rotation matrix
    rot_matrix = jnp.vstack([x_axis, y_axis, z_axis]).T

    # Decompose the rotation matrix to get Euler angles
    # ZYX convention: alpha, beta, gamma
    beta = jnp.arcsin(-rot_matrix[2, 0])
    alpha = jnp.arctan2(
        rot_matrix[1, 0] / jnp.cos(beta), rot_matrix[0, 0] / jnp.cos(beta)
    )
    gamma = jnp.arctan2(
        rot_matrix[2, 1] / jnp.cos(beta), rot_matrix[2, 2] / jnp.cos(beta)
    )

    return jnp.rad2deg(jnp.array([alpha, beta, gamma]))
