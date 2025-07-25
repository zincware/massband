import itertools
import jax.numpy as jnp
from ase.data import atomic_numbers
from jax import jit, vmap
from functools import partial


def generate_sorted_pairs(structure_names):
    """
    Generate and sort unique element/compound name pairs based on chemical priority.

    The sorting logic follows these rules:
    1. Element pairs are sorted by atomic number.
    2. Element-compound pairs prioritize the element and sort by atomic number then name.
    3. Compound-compound pairs are sorted alphabetically.

    Parameters
    ----------
    structure_names : list of str
        List of element and/or compound names.

    Returns
    -------
    list of tuple of str
        Sorted list of unique name pairs.

    Examples
    --------
    >>> structure_names = ['H', 'O', 'H2O', 'C']
    >>> generate_sorted_pairs(structure_names)
    [('C', 'H'), ('C', 'O'), ('H', 'O'), ('H', 'H2O'), ('O', 'H2O'), ('H2O', 'H2O')]
    """

    def is_element(name):
        return name in atomic_numbers

    def sort_key(pair):
        a, b = pair
        a_is_elem = is_element(a)
        b_is_elem = is_element(b)
        if a_is_elem and b_is_elem:
            return (
                0,
                min(atomic_numbers[a], atomic_numbers[b]),
                max(atomic_numbers[a], atomic_numbers[b]),
            )
        elif a_is_elem:
            return (1, atomic_numbers[a], b)
        elif b_is_elem:
            return (1, atomic_numbers[b], a)
        else:
            return (2, min(a, b), max(a, b))

    # Generate combinations with replacement
    pairs = list(itertools.combinations_with_replacement(structure_names, 2))

    # Sort elements within each pair to ensure canonical order
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

    # Remove duplicates and sort final list
    pairs = sorted(set(pairs), key=sort_key)

    return pairs


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


@partial(jit, static_argnames=["exclude_self"])
def mic_distances(frame_a, frame_b, frame_cell, exclude_self):
    inv_cell = jnp.linalg.inv(frame_cell.T)
    frac_a = frame_a @ inv_cell
    frac_b = frame_b @ inv_cell

    delta_frac = frac_a[:, None, :] - frac_b[None, :, :]
    delta_frac -= jnp.round(delta_frac)
    delta_cart = delta_frac @ frame_cell.T
    dists = jnp.linalg.norm(delta_cart, axis=-1)
    if exclude_self:
        dists = dists.at[jnp.diag_indices_from(dists)].set(
            jnp.inf
        )  # ignore self-distance

    return dists


@partial(jit, static_argnames=["exclude_self"])
def rdf_single_frame(pos_a, pos_b, frame_cell, exclude_self, bin_edges):
    dists = mic_distances(pos_a, pos_b, frame_cell, exclude_self=exclude_self)
    dists = dists.flatten()
    hist = jnp.histogram(dists, bins=bin_edges)[0]
    return hist


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

    # Process frames in batches to reduce memory usage
    hist_sum = jnp.zeros(len(bin_edges) - 1)

    # Create progress bar for batch processing
    n_batches = (n_frames + batch_size - 1) // batch_size

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_frames)

        # Extract batch data
        batch_pos_a = positions_a[start_idx:end_idx]
        batch_pos_b = positions_b[start_idx:end_idx]
        batch_cell = cell[start_idx:end_idx]
        # TODO: cell is NVT, so it should always be the same for all frames
        # Compute RDF for this batch
        batch_histograms = vmap(rdf_single_frame, in_axes=(0, 0, 0, None, None))(
            batch_pos_a, batch_pos_b, batch_cell, exclude_self, bin_edges
        )

        # Accumulate histogram
        hist_sum += jnp.sum(batch_histograms, axis=0)

    # Normalize RDF
    volume = jnp.linalg.det(cell)
    min_box_length = jnp.min(jnp.linalg.norm(cell, axis=-1))  # across frames
    shell_volume = _ideal_gas_correction(bin_edges, L=min_box_length)

    number_density = positions_b.shape[1] / jnp.mean(volume)  # mean number density
    norm_factor = positions_a.shape[1] * number_density * shell_volume * n_frames

    g_r = hist_sum / norm_factor
    return g_r