import  zntrack
import znh5md
from laufband import Laufband
import ase
import jax.numpy as jnp

import jax.numpy as jnp
from jax import vmap


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

class RadialDistributionFunction(zntrack.Node):
    """
    Class to represent a radial distribution function (RDF) in a molecular dynamics simulation.
    """
    file: str = zntrack.deps_path()

    batch_size: int = zntrack.params()  # You can set a default or make it configurable

    def run(self):
        io = znh5md.IO(self.file)
        size = len(io)
        batch_start_index = list(range(0, size, self.batch_size))
        worker = Laufband(batch_start_index)
        for start in worker:
            end = min(start + self.batch_size, size)
            batch = io[start:end]
            rdf = self.compute_rdf(batch)
        

    def compute_rdf(self, batch: list[ase.Atoms]) -> dict[tuple[int, int], list]:
        # return the RDF for each permutation of the species, e.g. for H2O have O-O, O-H, H-H where 
        # Convert ASE objects to JAX arrays
        positions = jnp.stack([jnp.array(atoms.positions) for atoms in batch])  # (N, i, 3)
        cells = jnp.stack([jnp.array(atoms.cell[:]) for atoms in batch])  # (N, 3, 3)
        species = jnp.stack([jnp.array(atoms.get_chemical_symbols()) for atoms in batch])  # (N, i)

        N_frames, N_atoms, _ = positions.shape
        distances = compute_mic_distances(positions, cells)  # (N, i, i)

        # Flatten upper triangle distances (i < j), exclude self-distances (i == j)
        i, j = jnp.triu_indices(N_atoms, k=1)
        pairwise_dists = distances[:, i, j].reshape(-1)  # shape: (N_frames * num_pairs)

        # Define histogram bins
        r_max = 10.0
        bin_width = 0.1
        bin_edges = jnp.arange(0, r_max + bin_width, bin_width)  # includes right edge
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Histogram the distances
        hist = jnp.histogram(pairwise_dists, bins=bin_edges)[0]  # shape: (num_bins,)

        # Normalize RDF
        volume = jnp.mean(jnp.linalg.det(cells))  # average volume across frames
        number_density = (N_atoms * (N_atoms - 1) / 2) * N_frames / volume  # total pairs / volume

        shell_volumes = (4 / 3) * jnp.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)
        ideal_counts = number_density * shell_volumes  # expected count per shell
        g_r = hist / ideal_counts  # RDF

        return bin_centers, g_r
