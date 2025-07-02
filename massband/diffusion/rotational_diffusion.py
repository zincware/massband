import zntrack
from pathlib import Path
from massband.com import load_unwrapped_frames, identify_substructures, compute_com_trajectories
import jax.numpy as jnp
import numpy as np
from scipy.linalg import logm
from collections import defaultdict
from tqdm import tqdm

class RotationalSelfDiffusion(zntrack.Node):
    file: str | Path = zntrack.deps_path()
    sampling_rate: int = zntrack.params()
    timestep: float = zntrack.params()
    structures: list[str] | None = zntrack.params(default=None)

    diffusion_results: Path = zntrack.outs_path

    def run(self):
        frames, positions, cells = load_unwrapped_frames(self.file)
        masses = jnp.array(frames[0].get_masses())
        substructures = identify_substructures(frames[0], self.structures)

        com_positions = defaultdict(list)
        mol_orientations = defaultdict(list)  # This will store orientation vectors over time

        for structure, all_indices in substructures.items():
            for mol_indices in tqdm(all_indices):
                mol_masses = jnp.array([masses[i] for i in mol_indices])
                mol_positions = positions[:, mol_indices]
                mass_sum = jnp.sum(mol_masses)
                
                # Compute center of mass
                weighted_positions = mol_positions * mol_masses[None, :, None]
                com = jnp.sum(weighted_positions, axis=1) / mass_sum
                com_positions[structure].append(com)
                
                # Compute orientation vector
                # Step 1: Define reference orientation in first frame
                ref_positions = mol_positions[0] - com[0]  # Center positions
                if len(mol_indices) == 2:  # Diatomic molecule
                    # Orientation is simply the vector between the two atoms
                    orientation = ref_positions[1] - ref_positions[0]
                else:
                    # For more complex molecules, use principal axes of inertia
                    # Compute inertia tensor
                    displacements = ref_positions
                    inertia = jnp.zeros((3, 3))
                    for i in range(len(mol_indices)):
                        r = displacements[i]
                        inertia += mol_masses[i] * (jnp.eye(3) * jnp.sum(r**2) - jnp.outer(r, r))
                    
                    # Get principal axes (eigenvectors)
                    _, eigenvectors = jnp.linalg.eigh(inertia)
                    # Use the axis with smallest moment of inertia (most "oriented" axis)
                    orientation = eigenvectors[:, 0]
                
                # Normalize reference orientation
                ref_orientation = orientation / jnp.linalg.norm(orientation)
                
                # Step 2: Compute orientation in each frame
                orientations = []
                for t in range(positions.shape[0]):
                    current_positions = mol_positions[t] - com[t]  # Center positions
                    if len(mol_indices) == 2:
                        current_orientation = current_positions[1] - current_positions[0]
                    else:
                        # Recompute inertia tensor for current frame
                        displacements = current_positions
                        inertia = jnp.zeros((3, 3))
                        for i in range(len(mol_indices)):
                            r = displacements[i]
                            inertia += mol_masses[i] * (jnp.eye(3) * jnp.sum(r**2) - jnp.outer(r, r))
                        _, eigenvectors = jnp.linalg.eigh(inertia)
                        current_orientation = eigenvectors[:, 0]
                    
                    # Normalize and ensure consistent direction (avoid flips)
                    current_orientation = current_orientation / jnp.linalg.norm(current_orientation)
                    if jnp.dot(current_orientation, ref_orientation) < 0:
                        current_orientation = -current_orientation
                    
                    orientations.append(current_orientation)
                
                mol_orientations[structure].append(jnp.array(orientations))

        com_positions = {
            structure: jnp.stack(coms, axis=1) for structure, coms in com_positions.items()
        }
        mol_orientations = {
            structure: jnp.stack(orients, axis=1) for structure, orients in mol_orientations.items()
        }
