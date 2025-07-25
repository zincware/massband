import zntrack
from pathlib import Path
from massband.dataloader import TimeBatchedLoader

from typing import Literal, List, Tuple, Union
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem import Draw
import numpy as np
import matplotlib.pyplot as plt
from vesin import NeighborList
from tqdm import tqdm
import seaborn as sns

def select_atoms_flat_unique(
    mol: rdchem.Mol,
    smarts_or_smiles: str,
    hydrogens: Literal["include", "exclude", "isolated"] = "exclude",
) -> list[int]:
    """
    Selects a unique list of atom indices in a molecule using SMARTS or mapped SMILES.
    If the pattern contains atom maps (e.g., [C:1]), only the mapped atoms are returned.
    Otherwise, all atoms in the matched substructure are returned.
    
    Args:
        mol: RDKit molecule, which can contain explicit hydrogens.
        smarts_or_smiles: SMARTS (e.g., "[F]") or SMILES with atom maps (e.g., "C1[C:1]OC(=[O:1])O1").
        hydrogens: How to handle hydrogens in the final returned list.
    
    Returns:
        A single, flat list of unique integer atom indices matching the criteria.
    """
    patt = Chem.MolFromSmarts(smarts_or_smiles)
    
    if not patt:
        raise ValueError(f"Invalid SMARTS/SMILES: {smarts_or_smiles}")

    # Check if the pattern has any mapped atoms.
    mapped_pattern_indices = [
        atom.GetIdx() for atom in patt.GetAtoms() if atom.GetAtomMapNum() > 0
    ]
    
    matches = mol.GetSubstructMatches(patt)
    
    # 1. Get the core set of atoms. If the pattern is mapped, only use the
    #    indices corresponding to the mapped atoms. Otherwise, use all atoms.
    core_atom_indices = set()
    if mapped_pattern_indices:
        # Only collect indices from the molecule that correspond to a mapped atom in the pattern.
        for match_tuple in matches:
            for pattern_idx in mapped_pattern_indices:
                core_atom_indices.add(match_tuple[pattern_idx])
    else:
        # Original behavior: get all atoms from all matches if no maps are present.
        core_atom_indices = set(idx for match_tuple in matches for idx in match_tuple)

    # 2. Handle the `hydrogens` parameter based on this core set of atoms.
    if hydrogens == "include":
        final_indices = set(core_atom_indices)
        for idx in core_atom_indices:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetAtomicNum() != 1:
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetAtomicNum() == 1:
                        final_indices.add(neighbor.GetIdx())
        return sorted(list(final_indices))

    elif hydrogens == "exclude":
        heavy_only = {idx for idx in core_atom_indices if mol.GetAtomWithIdx(idx).GetAtomicNum() != 1}
        return sorted(list(heavy_only))

    elif hydrogens == "isolated":
        isolated_hydrogens = set()
        heavy_core_atoms = {idx for idx in core_atom_indices if mol.GetAtomWithIdx(idx).GetAtomicNum() != 1}
        for idx in heavy_core_atoms:
            atom = mol.GetAtomWithIdx(idx)
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 1:
                    isolated_hydrogens.add(neighbor.GetIdx())
        return sorted(list(isolated_hydrogens))
    
    return sorted(list(core_atom_indices))


def visualize_selected_molecules(mol: rdchem.Mol, a: list[int], b: list[int]):
    """
    Visualizes molecules that contain selected atoms, highlighting the selections.
    Duplicate molecular structures will only be plotted once.

    Args:
        mol: The RDKit molecule object, which may contain multiple fragments.
        a: A list of atom indices to be highlighted in the first color (e.g., pink).
        b: A list of atom indices to be highlighted in the second color (e.g., light blue).
    
    Returns:
        A PIL image object of the grid.
    """
    # Get separate molecule fragments from the main mol object
    frags = Chem.GetMolFrags(mol, asMols=True)
    frag_indices = Chem.GetMolFrags(mol, asMols=False)
    
    # --- Step 1: Collect all candidate molecules and their highlight data ---
    candidate_mols = []
    candidate_highlights = []
    candidate_colors = []
    
    all_selected_indices = set(a + b)

    # Define colors for highlighting
    color_a = (1.0, 0.7, 0.7) # Pink
    color_b = (0.7, 0.7, 1.0) # Light Blue

    for i, frag in enumerate(frags):
        original_indices_in_frag = set(frag_indices[i])
        
        # Check if this fragment contains any of the selected atoms
        if not all_selected_indices.isdisjoint(original_indices_in_frag):
            candidate_mols.append(frag)
            
            # Map original indices to the new indices within the fragment
            original_to_frag_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(frag_indices[i])}
            
            current_highlights = []
            current_colors = {}
            
            for idx in a:
                if idx in original_to_frag_map:
                    frag_idx = original_to_frag_map[idx]
                    current_highlights.append(frag_idx)
                    current_colors[frag_idx] = color_a

            for idx in b:
                if idx in original_to_frag_map:
                    frag_idx = original_to_frag_map[idx]
                    if frag_idx not in current_highlights:
                         current_highlights.append(frag_idx)
                    current_colors[frag_idx] = color_b # Color b takes precedence
            
            candidate_highlights.append(current_highlights)
            candidate_colors.append(current_colors)
            
    if not candidate_mols:
        print("No molecules to draw with the given selections.")
        return None

    # --- Step 2: Filter for unique molecules using canonical SMILES ---
    mols_to_draw = []
    highlight_lists = []
    highlight_colors = []
    seen_smiles = set()

    for i, candidate_mol in enumerate(candidate_mols):
        # Generate canonical SMILES to identify unique structures
        mol_no_hs = Chem.RemoveHs(candidate_mol)
        smi = Chem.MolToSmiles(mol_no_hs, canonical=True)
        
        if smi not in seen_smiles:
            seen_smiles.add(smi)
            mols_to_draw.append(candidate_mol)
            highlight_lists.append(candidate_highlights[i])
            highlight_colors.append(candidate_colors[i])

    # Draw the grid
    img = Draw.MolsToGridImage(
        mols_to_draw,
        molsPerRow=4,
        subImgSize=(200, 200),
        legends=[f'Molecule {i}' for i in range(len(mols_to_draw))],
        highlightAtomLists=highlight_lists,
        highlightAtomColors=highlight_colors
    )
    return img

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Literal
import jax.numpy as jnp
from massband.rdf.utils import compute_rdf

class SubstructureRadialDistributionFunction(zntrack.Node):
    """Calculate radial distribution functions for selected substructure pairs."""
    file: str | Path = zntrack.deps_path()
    structures: list[str] = zntrack.params(default_factory=list)
    pairs: list[tuple[str, str]] = zntrack.params(default_factory=list)
    hydrogens: list[tuple[Literal["include", "exclude", "isolated"], Literal["include", "exclude", "isolated"]]] = zntrack.params(default_factory=list)
    max_distance: float = zntrack.params(default=10.0)
    bin_width: float = zntrack.params(default=0.05)
    batch_size: int = zntrack.params(default=64)
    start: int = zntrack.params(default=0)
    stop: int | None = zntrack.params(default=None)
    step: int = zntrack.params(default=1)
    figures: Path = zntrack.outs_path(zntrack.nwd / "figures")
    rdf_results: dict = zntrack.outs()

    def plot_rdf_for_pairs(self, rdf_data):
        """
        Create RDF plots for each substructure pair.

        Args:
            rdf_data: Dict with RDF data for each pair.
        """
        # Create individual RDF plots for each pair
        for pair_idx, g_r in enumerate(rdf_data.values()):
            if len(g_r) == 0:
                print(f"Skipping RDF plot for pair {pair_idx}: No data.")
                continue
                
            # Create r-values from bin centers
            r_values = np.arange(len(g_r)) * self.bin_width + self.bin_width / 2.0
            
            plt.figure(figsize=(10, 6))
            plt.plot(r_values, g_r, linewidth=2, color='C0')
            plt.xlabel('r (Å)')
            plt.ylabel('g(r)')
            plt.title(f'Radial Distribution Function\n{self.pairs[pair_idx][0]} - {self.pairs[pair_idx][1]}')
            plt.grid(True, alpha=0.3)
            plt.xlim(0, min(self.max_distance, r_values[-1]))
            
            plot_path = self.figures / f'rdf_pair_{pair_idx}_{self.pairs[pair_idx][0]}_{self.pairs[pair_idx][1]}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved RDF plot for pair {pair_idx}: {plot_path}")

    def calculate_rdf_for_pairs(self, positions_dict, cell_array, pair_indices_list):
        """
        Calculate RDF for selected atom pairs using the selected substructures.

        Args:
            positions_dict: Dict with positions for each structure type over all frames.
            cell_array: Cell parameters for all frames.
            pair_indices_list: List of tuples containing (indices1, indices2) for each pair.
            
        Returns:
            A dictionary where keys are pair indices and values are RDF arrays.
        """
        # Create bin edges
        bin_edges = jnp.arange(0.0, self.max_distance + self.bin_width, self.bin_width)
        
        rdf_results = {}
        for pair_idx, (indices1, indices2) in enumerate(pair_indices_list):
            # Get positions for each substructure
            # Since we're working with substructures, we need to extract positions of selected atoms
            all_positions = positions_dict[list(positions_dict.keys())[0]]  # Get positions from first structure
            
            # Extract positions for selected atoms
            pos_a = all_positions[:, indices1, :]  # Shape: (n_frames, n_atoms_a, 3)
            pos_b = all_positions[:, indices2, :]  # Shape: (n_frames, n_atoms_b, 3)
            
            # Calculate RDF
            exclude_self = bool(set(indices1) & set(indices2))  # True if there's overlap
            g_r = compute_rdf(
                positions_a=pos_a,
                positions_b=pos_b,
                cell=cell_array,
                bin_edges=bin_edges,
                exclude_self=exclude_self
            )
            
            rdf_results[f"pair_{pair_idx}"] = g_r
            
        return rdf_results

    def run(self):
        self.figures.mkdir(parents=True, exist_ok=True)
        dl = TimeBatchedLoader(
            file=self.file,
            batch_size=self.batch_size,
            structures=self.structures,
            wrap=False,
            com=False,
            properties=["position", "cell"],
            start=self.start,
            stop=self.stop,
            step=self.step,
            map_to_dict=False,
        )
        print(dl.first_frame_chem)
        pair_indices = []
        for (smarts1, smarts2), (h1, h2) in zip(self.pairs, self.hydrogens):
            indices1 = select_atoms_flat_unique(dl.first_frame_chem, smarts1, hydrogens=h1)
            indices2 = select_atoms_flat_unique(dl.first_frame_chem, smarts2, hydrogens=h2)
            pair_indices.append((indices1, indices2))
            img = visualize_selected_molecules(dl.first_frame_chem, indices1, indices2)
            path = self.figures / f"{smarts1}_{smarts2}.png"
            idx = 0
            while Path(path).exists():
                path = self.figures / f"{smarts1}_{smarts2}_{idx}.png"
                idx += 1
            if img is not None:
                img.save(path)
        print("Pair indices:", pair_indices)
        
        # Collect all positions and cells for RDF calculation
        all_positions = []
        all_cells = []
        frame_counter = 0

        pbar = tqdm(dl, desc="Processing frames", total=dl.total_frames)

        for batch in pbar:
            if isinstance(batch, dict) and "position" in batch and "cell" in batch:
                pos_batch = batch["position"]
                cell_batch = batch["cell"]
                
                # Handle batch dimensions correctly
                if hasattr(pos_batch, 'shape') and len(pos_batch.shape) > 2:
                    # pos_batch is (batch_size, n_atoms, 3)
                    for i in range(pos_batch.shape[0]):
                        all_positions.append(pos_batch[i])
                        all_cells.append(cell_batch)
                        pbar.update(1)
                        frame_counter += 1
                else:
                    # pos_batch is (n_atoms, 3) - single frame
                    all_positions.append(pos_batch)
                    all_cells.append(cell_batch)
                    pbar.update(1)
                    frame_counter += 1
            else:
                # Legacy format or different structure
                all_positions.append(batch)
                all_cells.append(dl.first_frame_atoms.get_cell()[:])
                pbar.update(1)
                frame_counter += 1

        # Convert to jax arrays
        positions_array = jnp.array(all_positions)  # Shape: (n_frames, n_atoms, 3)
        cells_array = jnp.array(all_cells)  # Shape: (n_frames, 3, 3)
        
        # Create positions dict compatible with RDF calculation
        positions_dict = {"structure": positions_array}
        
        # Calculate RDF for all pairs
        rdf_data = self.calculate_rdf_for_pairs(positions_dict, cells_array, pair_indices)
        
        self.rdf_results = {
            "rdf_data": {k: v.tolist() for k, v in rdf_data.items()},
            "pairs": self.pairs,
            "bin_width": self.bin_width,
            "max_distance": self.max_distance
        }
        
        self.plot_rdf_for_pairs(rdf_data)