import itertools
import logging
from collections import defaultdict
from pathlib import Path

import ase
import jax.numpy as jnp
import rdkit2ase
import znh5md
import zntrack
from jax import vmap

from massband.rdf_plot import plot_rdf
from massband.utils import unwrap_positions, wrap_positions

log = logging.getLogger(__name__)


def compute_rdf(
    positions_a, positions_b, cell, bin_edges, batch_size=100, exclude_self=False
):
    """
    Compute the radial distribution function (RDF) g(r) between two sets of particles over time.

    Parameters
    ----------
    positions_a : jnp.ndarray
        Shape (n_frames, n_atoms_a, 3). Wrapped positions of group A.
    positions_b : jnp.ndarray
        Shape (n_frames, n_atoms_b, 3). Wrapped positions of group B.
    cell : jnp.ndarray
        Shape (n_frames, 3, 3). Cell matrix for each frame.
    bin_edges : jnp.ndarray
        Shape (n_bins + 1,). Bin edges for g(r).
    batch_size : int
        Batch size for vectorized RDF calculation over time (memory-performance tuning).

    Returns
    -------
    g_r : jnp.ndarray
        RDF values, shape (n_bins,)
    """
    n_frames = positions_a.shape[0]
    n_bins = len(bin_edges) - 1
    bin_width = bin_edges[1] - bin_edges[0]

    def mic_distances(frame_a, frame_b, frame_cell, exclude_self=True):
        inv_cell = jnp.linalg.inv(frame_cell.T)
        frac_a = frame_a @ inv_cell
        frac_b = frame_b @ inv_cell

        delta_frac = frac_a[:, None, :] - frac_b[None, :, :]
        delta_frac -= jnp.round(delta_frac)
        delta_cart = delta_frac @ frame_cell.T
        dists = jnp.linalg.norm(delta_cart, axis=-1)

        if exclude_self and frame_a.shape[0] == frame_b.shape[0]:
            dists = dists.at[jnp.diag_indices_from(dists)].set(
                jnp.inf
            )  # ignore self-distance

        return dists

    def rdf_single_frame(pos_a, pos_b, frame_cell):
        exclude_self = jnp.all(pos_a == pos_b)  # works if same object (intra-group)
        dists = mic_distances(pos_a, pos_b, frame_cell).flatten()
        hist = jnp.histogram(dists, bins=bin_edges)[0]
        return hist

    # Batch RDF evaluation over all frames
    histograms = vmap(rdf_single_frame)(positions_a, positions_b, cell)
    hist_sum = jnp.sum(histograms, axis=0)

    # Normalize RDF
    volume = jnp.linalg.det(cell)
    r_lower = bin_edges[:-1]
    r_upper = bin_edges[1:]
    shell_volume = (4 / 3) * jnp.pi * (r_upper**3 - r_lower**3)  # Shell volume per bin

    number_density = positions_b.shape[1] / jnp.mean(volume)  # mean number density
    norm_factor = positions_a.shape[1] * number_density * shell_volume * n_frames

    g_r = hist_sum / norm_factor
    return g_r


class RadialDistributionFunction(zntrack.Node):
    """
    Class to represent a radial distribution function (RDF) in a molecular dynamics simulation.
    """

    file: str = zntrack.deps_path()
    batch_size: int = zntrack.params()  # You can set a default or make it configurable
    bin_width: float = zntrack.params(0.05)  # Width of the bins for RDF
    structures: list[str] = zntrack.params()

    figures: Path = zntrack.outs_path(zntrack.nwd / "figures")

    def run(self):
        # TODO: need to ensure that in the first frame, all molecules are fully unwrapped!!
        io = znh5md.IO(self.file, variable_shape=False, include=["position", "box"])
        frames: list[ase.Atoms] = io[:]
        print(f"Loaded {len(frames)} frames from {self.file}")
        positions = jnp.stack([atoms.positions for atoms in frames])
        cells = jnp.stack([atoms.cell[:] for atoms in frames])
        masses = jnp.array(frames[0].get_masses())
        inv_cells = jnp.linalg.inv(cells)
        print(f"Positions shape: {positions.shape}, Cells shape: {cells.shape}")
        positions = jnp.transpose(positions, (1, 0, 2))
        positions = vmap(lambda x: unwrap_positions(x, cells, inv_cells))(positions)
        positions = jnp.transpose(positions, (1, 0, 2))
        print(f"Unwrapped positions shape: {positions.shape}")

        # TODO: all of this could also go to utils? E.g. a get_center_of_mass positions function
        substructures = defaultdict(list)
        # a dict of list[tuple[int, ...]]
        if self.structures:
            log.info(f"Searching for substructures in {len(self.structures)} patterns")
            for structure in self.structures:
                indices = rdkit2ase.match_substructure(
                    frames[0],
                    smiles=structure,
                    suggestions=self.structures,
                )
                if indices:
                    substructures[structure].extend(indices)
                    log.info(f"Found {len(indices)} matches for {structure}")

        # TODO: move to utils
        com_positions = defaultdict(list)

        for structure, all_indices in substructures.items():
            log.info(f"Computing COM positions for {structure}")

            for mol_indices in all_indices:
                mol_masses = jnp.array(
                    [masses[i] for i in mol_indices]
                )  # (n_atoms_in_mol,)
                mol_positions = positions[:, mol_indices]  # (n_frames, n_atoms_in_mol, 3)

                # Compute COM for each frame: weighted sum over atoms
                # Numerator: sum_i(m_i * r_i), Denominator: sum_i(m_i)
                mass_sum = jnp.sum(mol_masses)
                weighted_positions = (
                    mol_positions * mol_masses[None, :, None]
                )  # broadcast to (n_frames, n_atoms, 3)
                com = jnp.sum(weighted_positions, axis=1) / mass_sum  # (n_frames, 3)

                com_positions[structure].append(com)

        com_positions = {
            structure: jnp.stack(coms) for structure, coms in com_positions.items()
        }
        log.info(
            f"Found COM positions: { {k: v.shape for k, v in com_positions.items()} }"
        )

        # now wrap the compute the rdfs and wrap the positions

        rdfs = defaultdict(list)
        bin_edges = jnp.arange(
            0.0, 10.0 + self.bin_width, self.bin_width
        )  # You can customize max range
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        for struct_a, struct_b in itertools.combinations_with_replacement(
            com_positions.keys(), 2
        ):
            pos_a = com_positions[struct_a]  # shape (n_mols_a, n_frames, 3)
            pos_b = com_positions[struct_b]  # shape (n_mols_b, n_frames, 3)

            # Transpose for RDF: shape -> (n_frames, n_mols, 3)
            pos_a = jnp.transpose(pos_a, (1, 0, 2))
            pos_b = jnp.transpose(pos_b, (1, 0, 2))

            print(f"Computing RDF for {struct_a} - {struct_b}")
            print(f"Positions A shape: {pos_a.shape}, Positions B shape: {pos_b.shape}")

            # Wrap positions into the box
            wrapped_a = wrap_positions(pos_a, cells)
            wrapped_b = wrap_positions(pos_b, cells)

            # Compute RDF over all frames
            g_r = compute_rdf(
                positions_a=wrapped_a,
                positions_b=wrapped_b,
                cell=cells,
                bin_edges=bin_edges,
                batch_size=self.batch_size,
                exclude_self=True,  # Exclude self-distances for intra-group RDF
            )  # shape (n_bins,)

            rdfs[(struct_a, struct_b)].append(g_r)

        self.figures.mkdir(exist_ok=True, parents=True)
        plot_rdf(rdfs, self.figures / "rdf.png")
