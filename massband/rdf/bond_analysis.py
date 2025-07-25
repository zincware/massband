from pathlib import Path
from typing import Literal

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import zntrack
from tqdm import tqdm

from massband.dataloader import TimeBatchedLoader

from .utils import compute_rdf, select_atoms_flat_unique, visualize_selected_molecules


class SubstructureRadialDistributionFunction(zntrack.Node):
    """Calculate radial distribution functions for selected substructure pairs."""

    file: str | Path = zntrack.deps_path()
    structures: list[str] = zntrack.params(default_factory=list)
    pairs: list[tuple[str, str]] = zntrack.params(default_factory=list)
    hydrogens: list[
        tuple[
            Literal["include", "exclude", "isolated"],
            Literal["include", "exclude", "isolated"],
        ]
    ] = zntrack.params(default_factory=list)
    max_distance: float = zntrack.params(default=10.0)
    bin_width: float = zntrack.params(default=0.05)
    batch_size: int = zntrack.params(default=64)
    start: int = zntrack.params(default=0)
    stop: int | None = zntrack.params(default=None)
    step: int = zntrack.params(default=1)
    figures: Path = zntrack.outs_path(zntrack.nwd / "figures")
    results: dict = zntrack.outs()
    partial_number_densities: dict[str, float] = zntrack.outs()

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
            plt.plot(r_values, g_r, linewidth=2, color="C0")
            plt.xlabel("r (Å)")
            plt.ylabel("g(r)")
            plt.title(
                f"Radial Distribution Function\n{self.pairs[pair_idx][0]} - {self.pairs[pair_idx][1]}"
            )
            plt.grid(True, alpha=0.3)
            plt.xlim(0, min(self.max_distance, r_values[-1]))

            # Sanitize SMARTS patterns for safe filenames
            safe_smarts1 = (
                self.pairs[pair_idx][0]
                .replace("[", "")
                .replace("]", "")
                .replace(":", "_")
            )
            safe_smarts2 = (
                self.pairs[pair_idx][1]
                .replace("[", "")
                .replace("]", "")
                .replace(":", "_")
            )
            plot_path = (
                self.figures / f"rdf_pair_{pair_idx}_{safe_smarts1}_{safe_smarts2}.png"
            )
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
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
            all_positions = positions_dict[
                list(positions_dict.keys())[0]
            ]  # Get positions from first structure

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
                exclude_self=exclude_self,
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
            indices1 = select_atoms_flat_unique(
                dl.first_frame_chem, smarts1, hydrogens=h1
            )
            indices2 = select_atoms_flat_unique(
                dl.first_frame_chem, smarts2, hydrogens=h2
            )
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

        # Calculate partial number densities from first frame
        first_frame_atoms = dl.first_frame_atoms
        volume = first_frame_atoms.get_volume()  # In Å³
        self.partial_number_densities = {}

        for pair_idx, (indices1, indices2) in enumerate(pair_indices):
            smarts1, smarts2 = self.pairs[pair_idx]

            # Store number densities using the same keys as regular RDF (SMARTS patterns)
            # This ensures compatibility with CN and PMF nodes
            self.partial_number_densities[smarts1] = len(indices1) / volume
            self.partial_number_densities[smarts2] = len(indices2) / volume

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
                if hasattr(pos_batch, "shape") and len(pos_batch.shape) > 2:
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

        # Store results in the same format as regular RDF for compatibility
        results_formatted = {}
        for pair_idx, (smarts1, smarts2) in enumerate(self.pairs):
            pair_key = f"pair_{pair_idx}"
            if pair_key in rdf_data:
                # Use the same key format as regular RDF: "structure1|structure2"
                results_formatted[f"{smarts1}|{smarts2}"] = rdf_data[pair_key].tolist()

        # Store both internal format and RDF-compatible format
        self.results = results_formatted  # This matches the regular RDF.results format

        self.plot_rdf_for_pairs(rdf_data)
