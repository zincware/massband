import time
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import zntrack
from rdkit import Chem
from tqdm import tqdm

from massband.dataloader import TimeBatchedLoader
from massband.rdf.utils import select_atoms_flat_unique, visualize_selected_molecules


def _get_molecule_ids(mol: Chem.Mol) -> np.ndarray:
    """Creates an array mapping each atom index to a molecule ID."""
    mol_ids = np.zeros(mol.GetNumAtoms(), dtype=int)
    # GetMolFrags returns a tuple of tuples, e.g., ((0, 1, 2), (3, 4, 5))
    for mol_id, atom_indices in enumerate(Chem.GetMolFrags(mol, asMols=False)):
        mol_ids[list(atom_indices)] = mol_id
    return mol_ids


def _compute_vectors_pbc(pos_a, pos_b, cell):
    """Compute displacement vectors under periodic boundary conditions."""
    delta = pos_a - pos_b
    inv_cell = jnp.linalg.inv(cell)
    scaled_delta = delta @ inv_cell
    scaled_delta -= jnp.round(scaled_delta)
    return scaled_delta @ cell


@jax.jit
def _compute_bond_matrix(
    positions: jnp.ndarray,
    cell: jnp.ndarray,
    indices_a: jnp.ndarray,
    indices_b: jnp.ndarray,
    mol_ids_a: jnp.ndarray,
    mol_ids_b: jnp.ndarray,
    distance_cutoff: float,
    exclude_self: bool,
) -> jnp.ndarray:
    """
    Computes a boolean matrix indicating bond formation based on distance.

    Args:
        positions: Atom positions for the frame. (n_atoms, 3)
        cell: Simulation cell matrix. (3, 3)
        indices_a: Indices of atoms in the first group. (n_atoms_a,)
        indices_b: Indices of atoms in the second group. (n_atoms_b,)
        mol_ids_a: Molecule IDs for each atom in group A.
        mol_ids_b: Molecule IDs for each atom in group B.
        distance_cutoff: Max distance between atoms to be considered bonded.
        exclude_self: If True, all intramolecular pairs are ignored.
    """
    pos_a, pos_b = positions[indices_a], positions[indices_b]
    pos_a_exp, pos_b_exp = pos_a[:, jnp.newaxis, :], pos_b[jnp.newaxis, :, :]

    vec_ab = _compute_vectors_pbc(pos_a_exp, pos_b_exp, cell)
    dist_ab = jnp.linalg.norm(vec_ab, axis=-1)

    # Create a mask to identify pairs of atoms from the same molecule.
    # This will be True if atom_i and atom_j are in the same molecule.
    intramolecular_mask = (mol_ids_a[:, jnp.newaxis] == mol_ids_b[jnp.newaxis, :])

    # Use jax.lax.cond to apply the mask only when exclude_self is True.
    dist_ab = jax.lax.cond(
        exclude_self,
        # If True, set distances for all intramolecular pairs to infinity.
        lambda d: jnp.where(intramolecular_mask, jnp.inf, d),
        # If False, do nothing to the distances.
        lambda d: d,
        dist_ab
    )

    return dist_ab < distance_cutoff


class SubstructureBondLifetime(zntrack.Node):
    """Calculate bond lifetimes for selected substructure pairs."""

    # --- Inputs ---
    file: Path = zntrack.deps_path()
    structures: list[str] = zntrack.params(default_factory=list)
    pairs: list[tuple[str, str]] = zntrack.params(default_factory=list)
    hydrogens: list[
        tuple[
            Literal["include", "exclude", "isolated"],
            Literal["include", "exclude", "isolated"],
        ]
    ] = zntrack.params(default_factory=list)
    distance_cutoff: float = zntrack.params(3.5)
    batch_size: int = zntrack.params(64)
    start: int = zntrack.params(0)
    stop: int | None = zntrack.params(None)
    step: int = zntrack.params(1)
    timestep: float = zntrack.params(0.5)
    sampling_rate: int = zntrack.params(2000)

    # --- Outputs ---
    figures: Path = zntrack.outs_path(zntrack.nwd / "figures")
    results: dict = zntrack.outs()

    def _plot_results(self, lifetime_data):
        """Create and save plots for each pair's autocorrelation function."""
        self.figures.mkdir(parents=True, exist_ok=True)
        for pair_idx, data in lifetime_data.items():
            pair_smarts = self.pairs[pair_idx]
            time_axis_ps, autocorr, lifetime_ps = data["time_axis_ps"], data["autocorrelation"], data["lifetime_ps"]
            plt.figure(figsize=(10, 6)); plt.plot(time_axis_ps, autocorr, label=f"Lifetime τ = {lifetime_ps:.2f} ps")
            plt.xlabel("Time (ps)"); plt.ylabel("Bond Autocorrelation C(t)"); plt.title(f"Bond Lifetime: {pair_smarts[0]} – {pair_smarts[1]}")
            plt.legend(); plt.grid(True, linestyle="--", alpha=0.6); plt.xlim(left=0); plt.ylim(bottom=0)
            plot_path = self.figures / f"lifetime_pair_{pair_idx}_{pair_smarts[0]}_{pair_smarts[1]}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight"); plt.close()
            print(f"Saved lifetime plot to {plot_path}")

    def run(self):
        """Main execution method for the node."""
        dl = TimeBatchedLoader(
            file=self.file, batch_size=self.batch_size, structures=self.structures, wrap=False,
            properties=["position", "cell"], start=self.start, stop=self.stop, step=self.step,
            com=False, map_to_dict=False,
        )
        time_step_fs = self.timestep * self.sampling_rate

        print("Step 1: Selecting atoms and preparing molecule IDs...")
        # Create a map from atom index to molecule ID for the whole system
        molecule_ids = _get_molecule_ids(dl.first_frame_chem)
        
        all_pair_data = []
        for pair_idx, ((smarts1, smarts2), (h1, h2)) in enumerate(zip(self.pairs, self.hydrogens)):
            indices1 = select_atoms_flat_unique(dl.first_frame_chem, smarts1, hydrogens=h1)
            indices2 = select_atoms_flat_unique(dl.first_frame_chem, smarts2, hydrogens=h2)
            
            # Get the molecule IDs corresponding to the selected atoms
            mol_ids1 = molecule_ids[indices1]
            mol_ids2 = molecule_ids[indices2]
            
            exclude_self = smarts1 == smarts2
            if exclude_self:
                print(f"Pair {pair_idx}: Found {len(indices1)} atoms. All intramolecular interactions will be excluded.")
            else:
                print(f"Pair {pair_idx}: Found {len(indices1)} atoms for group 1 and {len(indices2)} for group 2.")
            
            all_pair_data.append(
                (jnp.array(indices1), jnp.array(indices2), jnp.array(mol_ids1), jnp.array(mol_ids2), exclude_self)
            )

        print(f"\nStep 2: Processing {dl.total_frames} frames to identify bonds...")
        all_bond_matrices = []
        pbar = tqdm(dl, total=dl.total_frames, desc="Identifying bonds")
        
        for batch in pbar:
            positions_batch, cell_batch = jnp.array(batch["position"]), jnp.array(batch["cell"])
            
            for i in range(positions_batch.shape[0]):
                positions_frame = positions_batch[i]
                cell_frame = cell_batch[i] if cell_batch.ndim == 4 else cell_batch

                frame_bonds = []
                for indices1, indices2, mol_ids1, mol_ids2, exclude_self in all_pair_data:
                    bond_matrix = _compute_bond_matrix(
                        positions_frame, cell_frame, indices1, indices2, mol_ids1, mol_ids2, 
                        self.distance_cutoff, exclude_self
                    )
                    frame_bonds.append(bond_matrix)
                all_bond_matrices.append(frame_bonds)

        print("\nStep 3: Calculating autocorrelation and lifetimes...")
        final_results = {}
        for pair_idx, _ in enumerate(self.pairs):
            bond_history = jnp.stack([frame[pair_idx] for frame in all_bond_matrices])
            num_frames = bond_history.shape[0]
            max_delay = num_frames // 2
            
            h_avg = jnp.mean(bond_history)
            if h_avg == 0:
                print(f"Warning: No bonds ever formed for pair {pair_idx}. Skipping.")
                autocorr, lifetime_ps = np.zeros(max_delay), 0.0
            else:
                autocorr = []
                for t in tqdm(range(max_delay), desc=f"Autocorrelation Pair {pair_idx}", leave=False):
                    h0_ht = bond_history[:num_frames - t] * bond_history[t:]
                    ct = jnp.mean(h0_ht) / h_avg
                    autocorr.append(ct)
                autocorr = np.array(autocorr)
                lifetime_fs = np.sum(autocorr) * time_step_fs
                lifetime_ps = lifetime_fs / 1000.0

            time_axis_ps = (np.arange(max_delay) * time_step_fs) / 1000.0
            final_results[pair_idx] = {
                "autocorrelation": autocorr.tolist(), "lifetime_ps": lifetime_ps.item(), "time_axis_ps": time_axis_ps.tolist()
            }
            print(f"Pair {pair_idx} Average Lifetime: {lifetime_ps:.3f} ps")

        self.results = final_results
        self._plot_results(final_results)