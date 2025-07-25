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

class BondAnalysis(zntrack.Node):
    """Analyze bonds in structures.
    
    Parameters
    ----------
    file : str | Path
        Path to the input file containing structure data.
    pairs: list[tuple[str, str]]
        The pairs of atoms to analyze. For a given structures like
        `["C1COC(=O)O1", "F[P-](F)(F)(F)(F)F"]`, the pairs could be
        `[("C1[C:1]OC(=O)O1", "[F:1][P-]([F:1])([F:1])([F:1][F:1])")]`.
    """
    file: str | Path = zntrack.deps_path()
    structures: list[str] = zntrack.params(default_factory=list)
    pairs: list[tuple[str, str]] = zntrack.params(default_factory=list)
    hydrogens: list[tuple[Literal["include", "exclude", "isolated"], Literal["include", "exclude", "isolated"]]] = zntrack.params(default_factory=list)
    bond_distance_threshold: float = zntrack.params(default=3.0)
    batch_size: int = zntrack.params(default=64)
    start: int = zntrack.params(default=0)
    stop: int|None = zntrack.params(default=None)
    step: int = zntrack.params(default=1)
    figures: Path = zntrack.outs_path(zntrack.nwd / "figures")
    bond_distances: dict = zntrack.outs()

    def plot_bond_distances(self, all_bond_distances, time_steps):
        """
        Create plots showing bond distance evolution and distribution for each pair.
        
        This function generates two plots per pair:
        1. The mean bond distance over time, with the standard deviation as a shaded area.
        2. A histogram and Kernel Density Estimate (KDE) of all bond distances found.
        
        Args:
            all_bond_distances: Dictionary with bond distances for each pair over time.
            time_steps: List of time step indices.
        """
        # Use a nicer plot style
        sns.set_theme(style="whitegrid")

        for pair_idx, (pair_key, distances_over_time) in enumerate(all_bond_distances.items()):
            
            # --- Plot 1: Mean and Standard Deviation over Time ---
            
            mean_distances = [np.mean(d) if d else np.nan for d in distances_over_time]
            std_distances = [np.std(d) if d else np.nan for d in distances_over_time]
            
            # Convert to numpy arrays for easier calculations
            mean_distances = np.array(mean_distances)
            std_distances = np.array(std_distances)

            plt.figure(figsize=(12, 7))
            
            # Plot the mean line
            plt.plot(time_steps, mean_distances, label='Mean Distance', color='C0')
            
            # Add the shaded standard deviation region
            plt.fill_between(
                time_steps, 
                mean_distances - std_distances, 
                mean_distances + std_distances, 
                color='C0', 
                alpha=0.2, 
                label='Std. Deviation'
            )
            
            plt.xlabel('Time Step')
            plt.ylabel('Bond Distance (Å)')
            plt.title(f'Mean Bond Distance Over Time\n{self.pairs[pair_idx][0]} to {self.pairs[pair_idx][1]}')
            plt.legend()
            plt.xlim(time_steps[0], time_steps[-1])

            # Save plot
            plot_path = self.figures / f'bond_distances_pair_{pair_idx}_time_evolution.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved time evolution plot for pair {pair_idx}: {plot_path}")

            # --- Plot 2: Overall Bond Length Distribution ---
            
            # Flatten the list of lists into a single list of all observed distances
            all_distances_flat = [
                distance for frame_distances in distances_over_time for distance in frame_distances
            ]

            if not all_distances_flat:
                print(f"Skipping distribution plot for pair {pair_idx}: No bonds found.")
                continue

            plt.figure(figsize=(10, 6))
            
            # Create a histogram with a Kernel Density Estimate (KDE) overlay
            sns.histplot(all_distances_flat, kde=True, stat="density", binwidth=0.05)
            
            plt.xlabel('Bond Distance (Å)')
            plt.ylabel('Density')
            plt.title(f'Overall Bond Distance Distribution\n{self.pairs[pair_idx][0]} to {self.pairs[pair_idx][1]}')
            
            # Save plot
            dist_plot_path = self.figures / f'bond_distances_pair_{pair_idx}_distribution.png'
            plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved distribution plot for pair {pair_idx}: {dist_plot_path}")

    def calculate_bond_distances(self, positions, cell, pair_indices_list):
        """
        Calculate bond distances for selected atom pairs using vesin.NeighborList.
        
        Args:
            positions: Array of atomic positions.
            cell: Unit cell parameters.
            pair_indices_list: List of tuples containing (indices1, indices2) for each pair.
            
        Returns:
            A dictionary with bond distances for each pair.
        """
        nl = NeighborList(cutoff=self.bond_distance_threshold, full_list=True)
        
        # 1. Request pair indices ('P') and distances ('d').
        #    According to the vesin API, this returns a list of arrays: [pairs, distances]
        results = nl.compute(
            points=positions, 
            box=cell, 
            periodic=True, 
            quantities='Pd'
        )
        
        # 2. Unpack the list of arrays by index.
        #    results[0] is the array of pairs, shape (n_pairs, 2)
        #    results[1] is the array of distances, shape (n_pairs,)
        all_pairs = results[0]
        all_distances = results[1]

        pair_distances = {}
        for pair_idx, (indices1, indices2) in enumerate(pair_indices_list):
            distances = []
            pair_key = f"pair_{pair_idx}"
            
            # Use sets for efficient O(1) membership checking.
            set1 = set(indices1)
            set2 = set(indices2)
            
            # Iterate through the computed neighbors once to find matching pairs.
            for (i, j), dist in zip(all_pairs, all_distances):
                # Check if one atom is in the first group and the other is in the second.
                if (i in set1 and j in set2) or (i in set2 and j in set1):
                    distances.append(dist)
            
            pair_distances[pair_key] = distances
        
        return pair_distances

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
                map_to_dict=False
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
        
        ## Initialize storage for bond distances over time
        all_bond_distances = {f"pair_{i}": [] for i in range(len(pair_indices))}
        time_steps = []
        
        # Keep track of the absolute frame index
        frame_counter = 0
        
        # Loop over BATCHES from the data loader
        for batch in tqdm(dl):
            pos_batch = batch["position"]
            cell_batch = batch["cell"]
            
            # Loop over each FRAME within the current batch
            for i in range(pos_batch.shape[0]):
                # Get data for a single frame
                single_pos = pos_batch[i]
                single_cell = cell_batch
                
                
                # Calculate bond distances for this single frame
                frame_distances = self.calculate_bond_distances(
                    single_pos, single_cell, pair_indices
                )
                
                # Store distances for each pair for this frame
                for pair_key, distances in frame_distances.items():
                    all_bond_distances[pair_key].append(distances)
                
                # Calculate and store the correct time step for this frame
                current_time_step = self.start + frame_counter * self.step
                time_steps.append(current_time_step)
                frame_counter += 1

        # Store results
        self.bond_distances = {
            "distances": all_bond_distances,
            "time_steps": time_steps,
            "pairs": self.pairs
        }
        
        # Generate plots for each pair
        self.plot_bond_distances(all_bond_distances, time_steps)