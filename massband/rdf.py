import itertools
import logging
from collections import defaultdict
from pathlib import Path

import jax.numpy as jnp
import zntrack
from ase.data import atomic_numbers
from jax import vmap

from massband.com import center_of_mass_trajectories
from massband.rdf_fit import FIT_METHODS
from massband.rdf_plot import plot_rdf
from massband.utils import wrap_positions

log = logging.getLogger(__name__)


def _ideal_gas_correction(bin_edges: jnp.ndarray, L: float) -> jnp.ndarray:
    """
    Compute corrected shell volumes for RDF normalization due to finite box size.

    Parameters
    ----------
    bin_edges : jnp.ndarray
        Bin edges for RDF, shape (n_bins + 1,)
    L : float
        Box length (assumed cubic for correction model)

    Returns
    -------
    shell_volumes : jnp.ndarray
        Corrected shell volumes for each bin, shape (n_bins,)
    """
    r_lower = bin_edges[:-1]
    r_upper = bin_edges[1:]
    r = 0.5 * (r_lower + r_upper)
    dr = r_upper - r_lower
    r_scaled = r / L

    def spherical(r):
        return 4 * jnp.pi * r**2

    def correction1(r_scaled):
        return 2 * jnp.pi * r_scaled * (3 - 4 * r_scaled) * L**2

    def correction2(r_scaled):
        sqrt_term = jnp.sqrt(4 * r_scaled**2 - 2)
        arctan_1 = jnp.arctan(sqrt_term)
        arctan_2 = (
            8
            * r_scaled
            * jnp.arctan(
                (2 * r_scaled * (4 * r_scaled**2 - 3))
                / (sqrt_term * (4 * r_scaled**2 + 1))
            )
        )
        return 2 * r_scaled * (3 * jnp.pi - 12 * arctan_1 + arctan_2) * L**2

    shell_area = jnp.where(
        r_scaled <= 0.5,
        spherical(r),
        jnp.where(
            r_scaled <= jnp.sqrt(2) / 2,
            correction1(r_scaled),
            correction2(r_scaled),
        ),
    )

    return shell_area * dr


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

    def rdf_single_frame(pos_a, pos_b, frame_cell, exclude_self):
        dists = mic_distances(
            pos_a, pos_b, frame_cell, exclude_self=exclude_self
        ).flatten()
        hist = jnp.histogram(dists, bins=bin_edges)[0]
        return hist

    # Batch RDF evaluation over all frames
    histograms = vmap(rdf_single_frame, in_axes=(0, 0, 0, None))(
        positions_a, positions_b, cell, exclude_self
    )
    hist_sum = jnp.sum(histograms, axis=0)

    # Normalize RDF
    volume = jnp.linalg.det(cell)
    min_box_length = jnp.min(jnp.linalg.norm(cell, axis=-1))  # across frames
    shell_volume = _ideal_gas_correction(bin_edges, L=min_box_length)

    number_density = positions_b.shape[1] / jnp.mean(volume)  # mean number density
    norm_factor = positions_a.shape[1] * number_density * shell_volume * n_frames

    g_r = hist_sum / norm_factor
    return g_r


class RadialDistributionFunction(zntrack.Node):
    """
    Class to represent a radial distribution function (RDF) in a molecular dynamics simulation.
    """

    file: str | Path = zntrack.deps_path()
    batch_size: int = zntrack.params()  # You can set a default or make it configurable
    bin_width: float = zntrack.params(0.05)  # Width of the bins for RDF
    structures: list[str] | None = zntrack.params()

    bayesian: bool = zntrack.params(False)  # Whether to use Bayesian fitting
    fit_method: FIT_METHODS = zntrack.params("gaussian")  # Method for fitting the

    results: dict[str, list[float]] = zntrack.outs()

    figures: Path = zntrack.outs_path(zntrack.nwd / "figures")

    def run(self):
        com_positions, cells = center_of_mass_trajectories(
            file=self.file, structures=self.structures, wrap=True
        )

        # now wrap the compute the rdfs and wrap the positions

        self.results = defaultdict(list)
        bin_edges = jnp.arange(
            0.0, 10.0 + self.bin_width, self.bin_width
        )  # You can customize max range

        def is_element(name):
            return name in atomic_numbers

        def sort_key(pair):
            a, b = pair
            a_is_elem = is_element(a)
            b_is_elem = is_element(b)
            if a_is_elem and b_is_elem:
                # Both elements: sort by atomic number
                return (
                    0,
                    min(atomic_numbers[a], atomic_numbers[b]),
                    max(atomic_numbers[a], atomic_numbers[b]),
                )
            elif a_is_elem:
                # a is element, b is not
                return (1, atomic_numbers[a], b)
            elif b_is_elem:
                # b is element, a is not
                return (1, atomic_numbers[b], a)
            else:
                # Neither is element: sort alphabetically
                return (2, min(a, b), max(a, b))

        pairs = list(itertools.combinations_with_replacement(com_positions.keys(), 2))
        pairs = [
            tuple(
                sorted(
                    pair,
                    key=lambda x: (
                        not is_element(x),
                        atomic_numbers.get(x, float("inf")),
                        x,
                    ),
                )
            )
            for pair in pairs
        ]
        pairs = sorted(set(pairs), key=sort_key)

        for struct_a, struct_b in pairs:
            pos_a = com_positions[struct_a]
            pos_b = com_positions[struct_b]

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
                exclude_self=(
                    struct_a == struct_b
                ),  # Exclude self-distances for intra-group RDF
            )  # shape (n_bins,)

            self.results[(struct_a, struct_b)].append(g_r)

        self.results = {k: jnp.array(v).mean(axis=0) for k, v in self.results.items()}
        self.figures.mkdir(exist_ok=True, parents=True)
        plot_rdf(
            self.results,
            self.figures / "rdf.png",
            bayesian=self.bayesian,
            fit_method=self.fit_method,
        )

        self.results = {"|".join(k): v.tolist() for k, v in self.results.items()}
