from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import zntrack
from tqdm import tqdm

from massband.abc import ComparisonResults

# from massband.com import identify_substructures, load_unwrapped_frames
# from massband.diffusion.utils import compute_msd_fft
from massband.rotation.utils import (
    _orientation_to_euler_single,
    compute_all_orientations,
    compute_com,
    compute_orientation,
    unwrap_angles,
)

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

    @classmethod
    def compare(cls, *nodes: "RotationalSelfDiffusion") -> ComparisonResults:
        """
        Compare rotational self-diffusion results from multiple runs.

        This method generates an overlay plot of the Mean Squared Angular
        Displacement (MSAD) curves for each common structure and angle
        (alpha, beta, gamma) found across the provided nodes.
        """
        figures = {}

        # 1. Identify base keys (e.g., "structure_angle") for each node
        all_base_keys = []
        for node in nodes:
            if not node.results:
                continue
            # Extract base keys by finding all '_msd' keys and removing the suffix
            node_base_keys = {
                key.replace("_msd", "") for key in node.results if key.endswith("_msd")
            }
            all_base_keys.append(node_base_keys)

        if not all_base_keys:
            return {"frames": [], "figures": {}}

        # 2. Find the common base keys across all nodes
        common_base_keys = set.intersection(*all_base_keys)

        # 3. Create a comparison plot for each common base key
        for base_key in common_base_keys:
            fig = go.Figure()

            for node in nodes:
                # Check if this node has the data (it should, due to intersection)
                msd_key = f"{base_key}_msd"
                time_key = f"{base_key}_time"
                if msd_key not in node.results or time_key not in node.results:
                    continue

                # Get the data
                msd_curve = node.results[msd_key]
                time_axis = node.results[time_key]

                # Add trace to the plot
                fig.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=msd_curve,
                        mode="lines",
                        name=f"{node.name}",
                    )
                )

            # 4. Style the plot
            structure, angle_name = base_key.rsplit("_", 1)
            fig.update_layout(
                title_text=f"Rotational MSD Comparison: {structure} - {angle_name}",
                xaxis_title_text="Time / ps",
                yaxis_title_text="Mean Squared Angular Displacement / rad²",
                legend_title_text="Compared Runs",
            )
            figures[f"rotational_msd_comparison_{base_key}"] = fig

        return {"frames": [], "figures": figures}
