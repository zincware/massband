from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import zntrack
from tqdm import tqdm

from massband.com import identify_substructures, load_unwrapped_frames
from massband.diffusion.utils import compute_msd_fft


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


# Vectorized version using vmap
orientation_to_euler_jax = jax.jit(jax.vmap(_orientation_to_euler_single))


class RotationalSelfDiffusion(zntrack.Node):
    file: str | Path = zntrack.deps_path()
    sampling_rate: int = zntrack.params()
    timestep: float = zntrack.params()
    structures: list[str] | None = zntrack.params(default=None)

    diffusion_results: Path = zntrack.outs_path

    figures: Path = zntrack.outs_path(zntrack.nwd / "figures")

    def run(self):
        frames, positions, cells = load_unwrapped_frames(self.file)
        masses = jnp.array(frames[0].get_masses())
        substructures = identify_substructures(frames[0], self.structures)

        # JIT-compiled functions
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

        # Vectorized version for all frames
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
                term1 = (mol_masses[None, :] * r_sq).sum(axis=1)[:, None, None] * eye[
                    None, :, :
                ]
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

        com_positions = defaultdict(list)
        mol_orientations = defaultdict(list)

        for structure, all_indices in substructures.items():
            for mol_indices in tqdm(all_indices, desc=f"Processing {structure}"):
                mol_masses = jnp.array([masses[i] for i in mol_indices])
                mol_positions = positions[:, mol_indices]

                # Compute COMs (vectorized)
                com = compute_com(mol_positions, mol_masses)
                com_positions[structure].append(com)

                # Compute reference orientation
                ref_positions = mol_positions[0] - com[0]
                ref_orientation = compute_orientation(ref_positions, mol_masses)

                # Compute all orientations (vectorized)
                orientations = compute_all_orientations(
                    mol_positions, mol_masses, ref_orientation
                )

                # Convert to Euler angles for each molecule using the JAX version
                euler_angles = orientation_to_euler_jax(orientations)
                mol_orientations[structure].append(euler_angles)

        com_positions = {
            structure: jnp.stack(coms, axis=1)
            for structure, coms in com_positions.items()
        }
        mol_orientations = {
            structure: jnp.stack(orients, axis=1)
            for structure, orients in mol_orientations.items()
        }

        # Ensure the figures directory exists
        self.figures.mkdir(parents=True, exist_ok=True)

        unwrap_vmap = jax.jit(jax.vmap(unwrap_angles, in_axes=1, out_axes=1))

        # Plot Euler angles for each structure and component
        for structure, euler_angles_rad in tqdm(
            mol_orientations.items(), desc="Processing Euler angles"
        ):
            alpha_raw, beta_raw, gamma_raw = (
                euler_angles_rad[:, :, 0],
                euler_angles_rad[:, :, 1],
                euler_angles_rad[:, :, 2],
            )

            # 4. UNWRAP angles for each component and molecule
            alpha_unwrapped = unwrap_vmap(alpha_raw)
            beta_unwrapped = unwrap_vmap(beta_raw)
            gamma_unwrapped = unwrap_vmap(gamma_raw)

            # Calculate and plot MSD for each unwrapped angle component
            for angle_data, angle_name in zip(
                [alpha_unwrapped, beta_unwrapped, gamma_unwrapped],
                ["alpha", "beta", "gamma"],
            ):
                fig, ax = plt.subplots(figsize=(10, 6))

                # Use the corrected, efficient MSD function
                msd_curve = compute_msd_fft(angle_data)

                # Time axis in picoseconds
                time_axis = (
                    jnp.arange(len(msd_curve)) * self.timestep * self.sampling_rate
                )

                ax.plot(time_axis, msd_curve, label=f"Average for {structure}")

                ax.set_title(f"Rotational Self-Diffusion: {structure} - {angle_name}")
                ax.set_xlabel("Time (ps)")
                ax.set_ylabel("Mean Squared Angular Displacement (rad²)")
                ax.legend()
                ax.grid(True, linestyle="--", alpha=0.6)
                fig.savefig(self.figures / f"{structure}_{angle_name}_msd.png", dpi=300)
                plt.close(fig)
