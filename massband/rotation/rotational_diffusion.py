from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import zntrack
from tqdm import tqdm

from massband.com import identify_substructures, load_unwrapped_frames
from massband.diffusion.utils import compute_msd_fft
from massband.rotation.utils import _orientation_to_euler_single, compute_all_orientations, compute_com, compute_orientation, unwrap_angles

# https://www.biorxiv.org/content/10.1101/2025.05.27.656261v1.full.pdf





# Vectorized version using vmap
orientation_to_euler_jax = jax.jit(jax.vmap(_orientation_to_euler_single))


class RotationalSelfDiffusion(zntrack.Node):
    file: str | Path = zntrack.deps_path()
    sampling_rate: int = zntrack.params()
    timestep: float = zntrack.params()
    structures: list[str] | None = zntrack.params(default=None)

    diffusion_results: Path = zntrack.outs_path

    figures: Path = zntrack.outs_path(zntrack.nwd / "figures")

    def run(self):  # noqa: C901
        frames, positions, cells = load_unwrapped_frames(self.file)
        masses = jnp.array(frames[0].get_masses())
        substructures = identify_substructures(frames[0], self.structures)

       

       

        # Vectorized version for all frames
       
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
                    jnp.arange(len(msd_curve)) * self.timestep * self.sampling_rate / 1000
                )

                ax.plot(time_axis, msd_curve, label=f"Average for {structure}")

                ax.set_title(f"Rotational Self-Diffusion: {structure} - {angle_name}")
                ax.set_xlabel("Time / ps")
                ax.set_ylabel("Mean Squared Angular Displacement / rad²")
                ax.legend()
                ax.grid(True, linestyle="--", alpha=0.6)
                fig.savefig(self.figures / f"{structure}_{angle_name}_msd.png", dpi=300)
                plt.close(fig)
