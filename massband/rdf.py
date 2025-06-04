import itertools
from collections import defaultdict

import ase
import jax.numpy as jnp
import matplotlib.pyplot as plt
import znh5md
import zntrack
from ase.data import chemical_symbols
from jax import vmap
from laufband import Laufband


def compute_mic_distances(positions: jnp.ndarray, cells: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the pairwise distance matrix for each frame, accounting for periodic boundary conditions
    using the minimum image convention (MIC).

    Parameters:
    - positions: (N, i, 3) array of atomic positions
    - cells: (N, 3, 3) array of unit cell matrices per frame

    Returns:
    - distances: (N, i, i) array of pairwise distances per frame
    """

    def mic_frame(pos: jnp.ndarray, cell: jnp.ndarray):
        """
        MIC-based distance computation for a single frame.

        pos: (i, 3)
        cell: (3, 3)
        returns: (i, i) distance matrix
        """
        inv_cell = jnp.linalg.inv(cell.T)  # (3, 3)
        frac = pos @ inv_cell  # (i, 3) fractional
        delta_frac = frac[:, None, :] - frac[None, :, :]  # (i, i, 3)
        delta_frac = delta_frac - jnp.round(delta_frac)  # wrap to [-0.5, 0.5)
        delta_cart = delta_frac @ cell.T  # back to Cartesian
        dists = jnp.linalg.norm(delta_cart, axis=-1)  # (i, i)
        return dists

    # Vectorize over frames (N)
    return vmap(mic_frame)(positions, cells)


def plot_rdf(rdfs: defaultdict):
    # find best number of subplots
    n_rdfs = len(rdfs)
    n_cols = 3
    n_rows = (n_rdfs + n_cols - 1) // n_cols  # Ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    for ax, ((a, b), g_r_list) in zip(axes, rdfs.items()):
        g_r_array = jnp.stack(g_r_list)
        g_r_mean = jnp.mean(g_r_array, axis=0)
        r = 0.5 * (jnp.arange(len(g_r_mean)) + 0.5) * 0.1
        ax.plot(r, g_r_mean, label=f"{chemical_symbols[a]}-{chemical_symbols[b]}")
        ax.set_xlabel("Distance r (Å)")
        ax.set_ylabel("g(r)")
        ax.set_title(
            f"Radial Distribution Function: {chemical_symbols[a]}-{chemical_symbols[b]}"
        )
        ax.legend()
        ax.grid(True)
    fig.tight_layout()
    # Save the figure
    fig.savefig("rdf_plot.png")
    plt.close(fig)  # Close the figure to free memory


class RadialDistributionFunction(zntrack.Node):
    """
    Class to represent a radial distribution function (RDF) in a molecular dynamics simulation.
    """

    file: str = zntrack.deps_path()
    batch_size: int = zntrack.params()  # You can set a default or make it configurable
    bin_width: float = zntrack.params(0.05)  # Width of the bins for RDF

    def run(self):
        io = znh5md.IO(self.file, variable_shape=False, include=["position", "box"])
        size = len(io)
        batch_start_index = list(range(0, size, self.batch_size))
        worker = Laufband(batch_start_index, cleanup=True)

        # Collect RDFs for each species pair over all batches
        rdfs_all = defaultdict(list)

        for start in worker:
            end = min(start + self.batch_size, size)
            batch = io[start:end]
            rdf_batch = self.compute_rdf(batch)
            for pair, (r, g_r) in rdf_batch.items():
                rdfs_all[pair].append(g_r)

        # Plot the results
        plot_rdf(rdfs_all)
        # TODO: use rdkit2ase to map substructures

    def compute_rdf(
        self, batch: list[ase.Atoms]
    ) -> dict[tuple[int, int], tuple[jnp.ndarray, jnp.ndarray]]:
        # Convert ASE objects to JAX arrays
        positions = jnp.stack(
            [jnp.array(atoms.positions) for atoms in batch]
        )  # (N, i, 3)
        cells = jnp.stack([jnp.array(atoms.cell[:]) for atoms in batch])  # (N, 3, 3)
        atomic_numbers = jnp.array(batch[0].get_atomic_numbers())  # (i,)

        N_frames, N_atoms, _ = positions.shape
        distances = compute_mic_distances(positions, cells)  # (N, i, i)

        # Only take upper triangle (i < j)
        i, j = jnp.triu_indices(N_atoms, k=1)  # (n_pairs,)
        species_i = atomic_numbers[i]  # (n_pairs,)
        species_j = atomic_numbers[j]  # (n_pairs,)

        pairwise_dists = distances[:, i, j].reshape(-1)  # (N_frames * n_pairs,)

        # Define bins
        r_max = 10.0
        bin_edges = jnp.arange(0, r_max + self.bin_width, self.bin_width)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        shell_volumes = (4 / 3) * jnp.pi * (bin_edges[1:] ** 3 - bin_edges[:-1] ** 3)

        # Volume per frame (averaged)
        volume = jnp.mean(jnp.linalg.det(cells))

        # Unique atomic numbers
        unique_species = set(atomic_numbers.tolist())

        rdf_by_species = {}

        for a, b in itertools.combinations_with_replacement(sorted(unique_species), 2):
            # Mask for matching pairs (upper triangle only)
            frame_mask = ((species_i == a) & (species_j == b)) | (
                (species_i == b) & (species_j == a)
            )  # shape: (n_pairs,)
            frame_mask = jnp.asarray(frame_mask)

            # Repeat this mask for each frame
            full_mask = jnp.tile(frame_mask, N_frames)  # shape: (N_frames * n_pairs,)
            selected_dists = pairwise_dists[full_mask]

            if selected_dists.size == 0:
                continue  # skip empty

            # Histogram
            hist = jnp.histogram(selected_dists, bins=bin_edges)[0]

            # Estimate number density of this pair type
            n_pair_per_frame = frame_mask.sum()
            number_density = (n_pair_per_frame * N_frames) / volume

            ideal_counts = number_density * shell_volumes
            g_r = hist / ideal_counts

            rdf_by_species[(int(a), int(b))] = (bin_centers, g_r)

        return rdf_by_species
