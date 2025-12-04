import itertools
import logging
import typing as t
from functools import partial
from pathlib import Path

import ase
import jax.lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pint
import plotly.graph_objects as go
import rdkit2ase
import znh5md
import zntrack
from ase.data import atomic_numbers
from jax import jit, vmap
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from massband.abc import ComparisonResults
from massband.comparison_utils import (
    create_bar_comparison,
    create_difference_plot,
    create_overlay_plot,
)
from massband.utils import sanitize_structure_name

log = logging.getLogger(__name__)


class RDFData(t.TypedDict):
    """A data structure for storing RDF statistics."""

    bin_centers: list[float]
    g_r: list[float]
    g_r_std: list[float] | None
    g_r_ensemble: list[list[float]] | None
    unit: str
    number_density_a: float
    number_density_b: float


def generate_sorted_pairs(structure_names: list[str]) -> list[tuple[str, str]]:
    """
    Generate and sort unique element/compound name pairs based on chemical priority.

    The sorting creates a canonical order (e.g., H-O, not O-H; Element-Compound,
    not Compound-Element) to avoid duplicate calculations.

    The sorting logic follows these rules:
    1. Element pairs are sorted by atomic number.
    2. Element-compound pairs prioritize the element.
    3. Compound-compound pairs are sorted alphabetically.
    """

    def is_element(name):
        return name in atomic_numbers

    def sort_key(pair):
        a, b = pair
        a_is_elem, b_is_elem = is_element(a), is_element(b)

        # Canonicalize order within the pair before determining the sort key
        if (
            (a_is_elem and b_is_elem and atomic_numbers[a] > atomic_numbers[b])
            or (not a_is_elem and b_is_elem)
            or (not a_is_elem and not b_is_elem and a > b)
        ):
            a, b = b, a

        if a_is_elem and b_is_elem:
            return 0, atomic_numbers[a], atomic_numbers[b]
        elif a_is_elem:
            return 1, atomic_numbers[a], b
        # This case is implicitly handled by the swap above
        # elif b_is_elem:
        #     return (1, atomic_numbers[b], a)
        else:
            return 2, a, b

    # Generate all unique pairs and sort them canonically
    pairs = list(itertools.combinations_with_replacement(structure_names, 2))
    return sorted(pairs, key=sort_key)


def _finite_size_corrected_shell_volume(
    bin_edges: jnp.ndarray, L: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute corrected shell volumes for RDF normalization due to finite box size.

    This uses a simplified correction assuming a cubic box of side length L.
    For non-cubic boxes, L should be the minimum box dimension.
    """
    r_lower = bin_edges[:-1]
    r_upper = bin_edges[1:]
    r = 0.5 * (r_lower + r_upper)
    dr = r_upper - r_lower

    L_mean = jnp.mean(L)
    r_scaled = r / L_mean

    def spherical_shell(r_val):
        return 4 * jnp.pi * r_val**2

    shell_area = jnp.where(
        r_scaled <= 0.5,
        spherical_shell(r),
        4 * jnp.pi * r**2 * (1 - (3 / 2) * r_scaled + (1 / 2) * r_scaled**3),
    )
    # Set volume to zero for radii where the sphere intersects the box corners
    shell_area = jnp.where(r_scaled > jnp.sqrt(3) / 2, 0.0, shell_area)

    return shell_area * dr


@partial(jit, static_argnames=["exclude_self"])
def mic_distances(frame_a, frame_b, frame_cell, exclude_self: bool):
    """Compute minimum image convention distances between two sets of points."""
    inv_cell = jnp.linalg.inv(frame_cell)
    # Note: ASE cell vectors are rows, so pos @ inv(cell) gives fractional coords.
    frac_a = frame_a @ inv_cell
    frac_b = frame_b @ inv_cell

    delta_frac = frac_a[:, None, :] - frac_b[None, :, :]
    delta_frac -= jnp.round(delta_frac)
    delta_cart = delta_frac @ frame_cell
    dists = jnp.linalg.norm(delta_cart, axis=-1)

    if exclude_self:
        # Avoids JIT errors with dynamic sizes by checking if it's a square matrix
        if dists.shape[0] == dists.shape[1]:
            dists = dists.at[jnp.diag_indices_from(dists)].set(jnp.inf)

    return dists


@partial(jit, static_argnames=["exclude_self"])
def rdf_hist_single_frame(pos_a, pos_b, frame_cell, exclude_self, bin_edges):
    """Compute distance histogram for a single frame."""
    dists = mic_distances(pos_a, pos_b, frame_cell, exclude_self=exclude_self)
    return jnp.histogram(dists.flatten(), bins=bin_edges)[0]


def _compute_frame_histograms(
    positions_a: jnp.ndarray,
    positions_b: jnp.ndarray,
    cell: jnp.ndarray,
    bin_edges: jnp.ndarray,
    exclude_self: bool,
    batch_size: int,
    progress: Progress | None = None,
    batch_task_id: int | None = None,
) -> jnp.ndarray:
    """
    Compute distance histograms for all frames with batching.

    Parameters
    ----------
    positions_a : jnp.ndarray
        Positions of species A, shape (n_frames, n_particles_a, 3)
    positions_b : jnp.ndarray
        Positions of species B, shape (n_frames, n_particles_b, 3)
    cell : jnp.ndarray
        Simulation cell, shape (n_frames, 3, 3)
    bin_edges : jnp.ndarray
        Bin edges for histogram
    exclude_self : bool
        Whether to exclude self-interactions
    batch_size : int
        Number of frames to process per batch
    progress : Progress, optional
        Rich Progress instance for tracking progress
    batch_task_id : int, optional
        Task ID for updating batch progress

    Returns
    -------
    hist_per_frame : jnp.ndarray
        Histogram per frame, shape (n_frames, n_bins)
    """
    n_frames = positions_a.shape[0]
    hist_per_frame = []

    n_batches = (n_frames + batch_size - 1) // batch_size
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_frames)

        batch_pos_a = positions_a[start_idx:end_idx]
        batch_pos_b = positions_b[start_idx:end_idx]
        batch_cell = cell[start_idx:end_idx]

        # Vectorize histogram calculation over batch
        batch_histograms = vmap(rdf_hist_single_frame, in_axes=(0, 0, 0, None, None))(
            batch_pos_a, batch_pos_b, batch_cell, exclude_self, bin_edges
        )
        hist_per_frame.append(batch_histograms)

        # Update batch progress
        if progress is not None and batch_task_id is not None:
            progress.update(batch_task_id, advance=1)

    return jnp.concatenate(hist_per_frame, axis=0).astype(jnp.float32)


def _apply_cea_reweighting(
    hist_per_frame: jnp.ndarray,
    energy_model: jnp.ndarray,
    energy_mean: jnp.ndarray,
    temperature: float,
) -> jnp.ndarray:
    """
    Apply Cumulant Expansion Approximation (CEA) reweighting to histograms.

    The CEA formula: O_CEA = <O> - <O*h> + <O>*<h>
    where h = beta * (E_model - E_mean)

    This method propagates ensemble energy uncertainties to RDF uncertainties
    and is related to umbrella sampling reweighting methods.

    Parameters
    ----------
    hist_per_frame : jnp.ndarray
        Histogram per frame, shape (n_frames, n_bins)
    energy_model : jnp.ndarray
        Energy for specific model, shape (n_frames,)
    energy_mean : jnp.ndarray
        Mean energy across ensemble, shape (n_frames,)
    temperature : float
        Temperature in Kelvin

    Returns
    -------
    hist_cea : jnp.ndarray
        CEA-reweighted histogram (per-frame average), shape (n_bins,)

    References
    ----------
    Torrie, Glenn M., and John P. Valleau. "Nonphysical sampling distributions
    in Monte Carlo free-energy estimation: Umbrella sampling."
    Journal of Computational Physics 23.2 (1977): 187-199.
    https://doi.org/10.1016/0021-9991(77)90121-8
    """
    # Compute beta = 1/(k_B * T) in eV^-1
    ureg = pint.UnitRegistry()
    k_B = ureg.boltzmann_constant
    beta = (1.0 / (k_B * temperature * ureg.kelvin)).to("1/eV").magnitude

    # Compute CEA reweighting factor
    cea_factor = beta * (energy_model - energy_mean)  # (n_frames,)

    # Apply CEA formula
    O_mean = jnp.mean(hist_per_frame, axis=0)
    Oh_mean = jnp.mean(hist_per_frame * cea_factor[:, None], axis=0)
    h_mean = jnp.mean(cea_factor)

    hist_cea = O_mean - Oh_mean + O_mean * h_mean

    return hist_cea


def _normalize_histogram_to_rdf(
    hist: jnp.ndarray,
    n_a: int,
    n_b: int,
    cell: jnp.ndarray,
    bin_edges: jnp.ndarray,
    exclude_self: bool,
) -> tuple[jnp.ndarray, float, float]:
    """
    Normalize histogram to radial distribution function g(r).

    Applies finite-size corrections and computes number densities.

    Parameters
    ----------
    hist : jnp.ndarray
        Histogram of distances (can be per-frame average or accumulated)
    n_a : int
        Number of particles of species A
    n_b : int
        Number of particles of species B
    cell : jnp.ndarray
        Simulation cell, shape (n_frames, 3, 3)
    bin_edges : jnp.ndarray
        Bin edges for histogram
    exclude_self : bool
        Whether self-interactions are excluded (for same-species RDF)

    Returns
    -------
    g_r : jnp.ndarray
        Normalized radial distribution function
    number_density_a : float
        Number density of species A
    number_density_b : float
        Number density of species B
    """
    volume = jnp.linalg.det(cell)
    mean_volume = jnp.mean(volume)

    # Use minimum box dimension for finite-size correction
    min_box_lengths = jnp.min(jnp.linalg.norm(cell, axis=2), axis=1)
    shell_volume = _finite_size_corrected_shell_volume(bin_edges, L=min_box_lengths)

    # Compute number densities
    number_density_a = n_a / mean_volume
    number_density_b = n_b / mean_volume

    # Compute pair density (pairs per unit volume)
    if exclude_self:
        # For same-species RDF, we have N*(N-1) pairs per frame
        pair_density = (n_a * (n_a - 1)) / mean_volume
    else:
        # For different species, we have N_a * N_b pairs per frame
        pair_density = (n_a * n_b) / mean_volume

    # Normalize: g(r) = hist / (pair_density * shell_volume)
    norm_factor = pair_density * shell_volume

    # Avoid division by zero for empty bins or zero-volume shells
    g_r = jnp.where(norm_factor > 1e-9, hist / norm_factor, 0)

    return g_r, number_density_a, number_density_b


def compute_rdf(
    positions_a: jnp.ndarray,
    positions_b: jnp.ndarray,
    cell: jnp.ndarray,
    bin_edges: jnp.ndarray,
    batch_size: int = 100,
    exclude_self: bool = False,
    energy_ensemble: jnp.ndarray | None = None,
    temperature: float | None = None,
    progress: Progress | None = None,
    batch_task_id: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None, float, float]:
    """
    Compute radial distribution function (RDF) g(r) with optional ensemble uncertainty.

    When energy_ensemble is provided, computes uncertainty-aware RDFs using the
    Cumulant Expansion Approximation (CEA) method from i-pi committee reweighting.
    Otherwise, computes standard averaged RDF.

    Parameters
    ----------
    positions_a : jnp.ndarray
        Positions of species A, shape (n_frames, n_particles_a, 3)
    positions_b : jnp.ndarray
        Positions of species B, shape (n_frames, n_particles_b, 3)
    cell : jnp.ndarray
        Simulation cell, shape (n_frames, 3, 3)
    bin_edges : jnp.ndarray
        Bin edges for histogram, shape (n_bins + 1,)
    batch_size : int, default 100
        Number of frames to process per batch
    exclude_self : bool, default False
        Whether to exclude self-interactions (for same-species RDF)
    energy_ensemble : jnp.ndarray, optional
        Ensemble energies, shape (n_frames, n_models). If provided, enables
        uncertainty quantification via CEA method. Requires temperature parameter.
    temperature : float, optional
        Temperature in Kelvin. Required when energy_ensemble is provided.
    progress : Progress, optional
        Rich Progress instance for tracking progress
    batch_task_id : int, optional
        Task ID for updating batch progress

    Returns
    -------
    g_r : jnp.ndarray
        Mean RDF g(r), shape (n_bins,)
    g_r_std : jnp.ndarray or None
        Standard deviation of RDF across ensemble models if energy_ensemble provided,
        otherwise None
    g_r_ensemble : jnp.ndarray or None
        Per-model RDFs, shape (n_models, n_bins), if energy_ensemble provided,
        otherwise None
    number_density_a : float
        Number density of species A
    number_density_b : float
        Number density of species B

    References
    ----------
    Torrie, Glenn M., and John P. Valleau. "Nonphysical sampling distributions
    in Monte Carlo free-energy estimation: Umbrella sampling."
    Journal of Computational Physics 23.2 (1977): 187-199.
    https://doi.org/10.1016/0021-9991(77)90121-8

    Raises
    ------
    ValueError
        If energy_ensemble is provided but temperature is not
    """
    n_a = positions_a.shape[1]
    n_b = positions_b.shape[1]

    # Validate parameters for ensemble uncertainty
    if energy_ensemble is not None:
        if temperature is None:
            raise ValueError("temperature must be provided when energy_ensemble is set")
        n_frames, n_models = energy_ensemble.shape
    else:
        n_models = None

    # --- Standard RDF (no ensemble uncertainty) ---
    if energy_ensemble is None:
        # Compute per-frame histograms
        hist_per_frame = _compute_frame_histograms(
            positions_a,
            positions_b,
            cell,
            bin_edges,
            exclude_self,
            batch_size,
            progress,
            batch_task_id,
        )

        # Average over frames
        hist_avg = jnp.mean(hist_per_frame, axis=0)

        # Normalize to g(r)
        g_r, number_density_a, number_density_b = _normalize_histogram_to_rdf(
            hist_avg, n_a, n_b, cell, bin_edges, exclude_self
        )

        return g_r, None, None, number_density_a, number_density_b

    # --- Ensemble uncertainty RDF (CEA method) ---
    assert temperature is not None  # Already validated above
    assert n_models is not None

    # Compute mean energy per frame across ensemble
    energy_mean = jnp.mean(energy_ensemble, axis=1)  # (n_frames,)

    # Compute per-frame histograms once (shared across all models)
    # For first model: compute histograms with progress tracking
    n_frames = positions_a.shape[0]
    n_batches_per_model = (n_frames + batch_size - 1) // batch_size

    hist_per_frame = _compute_frame_histograms(
        positions_a,
        positions_b,
        cell,
        bin_edges,
        exclude_self,
        batch_size,
        progress,
        batch_task_id,
    )

    # Process each model in the ensemble
    g_r_models = []
    for model_idx in range(n_models):
        energy_model = energy_ensemble[:, model_idx]  # (n_frames,)

        # Apply CEA reweighting
        hist_cea = _apply_cea_reweighting(
            hist_per_frame, energy_model, energy_mean, temperature
        )

        # Normalize to g(r)
        g_r_model, number_density_a, number_density_b = _normalize_histogram_to_rdf(
            hist_cea, n_a, n_b, cell, bin_edges, exclude_self
        )

        g_r_models.append(g_r_model)

        # Update progress to match expected behavior (simulate per-model histogram computation)
        # Skip first model since histogram computation already updated progress
        if model_idx > 0 and progress is not None and batch_task_id is not None:
            progress.update(batch_task_id, advance=n_batches_per_model)

    # Compute ensemble statistics
    g_r_models = jnp.stack(g_r_models, axis=0)  # (n_models, n_bins)
    g_r_mean = jnp.mean(g_r_models, axis=0)
    g_r_std = jnp.std(g_r_models, axis=0)

    return g_r_mean, g_r_std, g_r_models, number_density_a, number_density_b


@partial(jit, static_argnames=["n_molecules", "n_atoms_per_mol"])
def _calculate_pbc_aware_com_for_frame(
    frame_positions, frame_cell, atom_indices, masses, n_molecules, n_atoms_per_mol
):
    """
    JIT-compiled function to compute PBC-aware wrapped COMs for all molecules in one frame.
    Uses a pseudo-space method to handle molecules split across periodic boundaries.
    This version correctly handles triclinic (non-orthogonal) cells.
    """
    mol_positions = frame_positions[atom_indices]

    # Convert to fractional coordinates using the inverse cell matrix
    inv_cell = jnp.linalg.inv(frame_cell)
    s_coords = mol_positions @ inv_cell

    def compute_com_dimension(s_coords_dim, masses_norm):
        # Map fractional coordinates [0, 1] to angles [0, 2*pi]
        theta = s_coords_dim * (2 * jnp.pi)
        # Compute the center of mass in this pseudo-space
        xi = jnp.cos(theta)
        zeta = jnp.sin(theta)
        xi_bar = jnp.sum(masses_norm * xi, axis=1)
        zeta_bar = jnp.sum(masses_norm * zeta, axis=1)
        # Convert the average angle back to a fractional coordinate
        theta_bar = jnp.arctan2(-zeta_bar, -xi_bar) + jnp.pi
        return theta_bar / (2 * jnp.pi)

    total_mass_per_mol = jnp.sum(masses)
    masses_norm = masses[None, :] / total_mass_per_mol
    masses_norm = jnp.broadcast_to(masses_norm, (n_molecules, len(masses)))

    # Compute COM for each dimension in fractional coordinates
    com_scaled = jnp.stack(
        [compute_com_dimension(s_coords[:, :, i], masses_norm) for i in range(3)],
        axis=1,
    )

    # Convert fractional COM back to Cartesian coordinates
    return com_scaled @ frame_cell


class RadialDistributionFunction(zntrack.Node):
    """
    Calculate radial distribution functions (RDFs) for molecular dynamics trajectories.

    Computes g(r) for atom-atom or center-of-mass pairs with proper periodic boundary
    condition handling and finite-size corrections.

    Parameters
    ----------
    data: znh5md.IO | list[ase.Atoms]| None, default None
        znh5md.IO object for trajectory data, as an alternative to 'file'.
    structures : list of str or None, default None
        SMILES strings for molecular structures. If provided, computes COM-COM RDFs.
        If None, computes atom-atom RDFs grouped by element.
    bin_width : float, default 0.05
        Width of distance bins in Angstrom.
    start : int, default 0
        Starting frame index.
    stop : int or None, default None
        Ending frame index. If None, processes all frames.
    step : int, default 1
        Frame sampling interval.
    batch_size : int, default 4
        Number of frames to process in each batch for memory management.
    uncertainties : str or None, default None
        Name of the property containing ensemble energies (e.g., "energy_ensemble").
        If provided, computes uncertainty-aware RDFs using the Cumulant Expansion
        Approximation (CEA) method. Requires temperature parameter.
    temperature : float or None, default None
        Temperature in Kelvin. Required when uncertainties is set.

    Attributes
    ----------
    rdf : dict
        Dictionary containing RDF data for each pair with keys:
        - 'bin_centers': List of bin center positions
        - 'g_r': List of mean RDF values
        - 'g_r_std': List of standard deviations (if uncertainties enabled)
        - 'g_r_ensemble': List of per-model RDF values (if uncertainties enabled)
        - 'unit': Distance unit string
        - 'number_density_a': Number density of species A
        - 'number_density_b': Number density of species B
    figures_path : Path
        Directory containing RDF plots.
    data_path : Path
        Directory containing RDF data files.
    """

    data: znh5md.IO | list[ase.Atoms] = zntrack.deps()
    structures: list[str] | None = zntrack.params(None)
    bin_width: float = zntrack.params(0.05)
    start: int = zntrack.params(0)
    stop: int | None = zntrack.params(None)
    step: int = zntrack.params(1)
    batch_size: int = zntrack.params(4)
    uncertainties: str | None = zntrack.params(None)
    temperature: float | None = zntrack.params(None)

    rdf: dict[str, RDFData] = zntrack.outs()
    figures_path: Path = zntrack.outs_path(zntrack.nwd / "figures")
    data_path: Path = zntrack.outs_path(zntrack.nwd / "data")

    @staticmethod
    def compare(  # noqa: C901
        *nodes: "RadialDistributionFunction",
        labels: list[str] | None = None,
        pairs: list[str] | None = None,
        use_plotly: bool = False,
    ) -> ComparisonResults:
        """Compare RDFs from multiple calculations.

        Parameters
        ----------
        *nodes : RadialDistributionFunction
            Multiple RDF node instances to compare
        labels : list[str] | None
            Labels for each node (e.g., ["ML potential", "DFT", "Experiment"]).
            If None, uses node indices as labels.
        pairs : list[str] | None
            Specific pairs to compare (e.g., ["Li-Li", "F-F"]).
            If None, compares all common pairs across all nodes.
        use_plotly : bool
            If True, creates plotly figures; otherwise matplotlib.

        Returns
        -------
        ComparisonResults
            Dictionary containing comparative RDF plots with keys:
            - "overlay_{pair}": overlay plot for each pair
            - "difference_{pair}": difference plot for each pair
            - "peak_positions": bar chart comparing first peak positions
            - "peak_heights": bar chart comparing first peak heights
            - "coordination_numbers": bar chart comparing coordination numbers
        """
        if len(nodes) < 2:
            raise ValueError("At least two nodes are required for comparison")

        # Generate default labels if not provided
        if labels is None:
            labels = [
                node.name if node.name else f"Node {i + 1}"
                for i, node in enumerate(nodes)
            ]
        elif len(labels) != len(nodes):
            raise ValueError(
                f"Number of labels ({len(labels)}) must match number of nodes ({len(nodes)})"
            )

        # Find common pairs across all nodes
        common_pairs = set(nodes[0].rdf.keys())
        for node in nodes[1:]:
            common_pairs &= set(node.rdf.keys())

        if not common_pairs:
            raise ValueError("No common pairs found across all nodes")

        # Filter to specific pairs if requested
        if pairs is not None:
            common_pairs = set(pairs) & common_pairs
            if not common_pairs:
                raise ValueError(
                    f"None of the requested pairs {pairs} are common to all nodes"
                )

        common_pairs = sorted(common_pairs)
        figures: dict[str, plt.Figure | go.Figure] = {}

        # Track peak statistics for all pairs
        peak_positions: dict[str, list[float]] = {}
        peak_heights: dict[str, list[float]] = {}
        coordination_numbers: dict[str, list[float]] = {}

        # Compare each pair
        for pair in common_pairs:
            # Extract data for this pair from all nodes
            x_data = []
            y_data = []
            uncertainties = []

            for node in nodes:
                rdf_data = node.rdf[pair]
                x = np.array(rdf_data["bin_centers"])
                y = np.array(rdf_data["g_r"])
                x_data.append(x)
                y_data.append(y)

                # Get uncertainties if available
                if rdf_data.get("g_r_std") is not None:
                    uncertainties.append(np.array(rdf_data["g_r_std"]))
                else:
                    uncertainties.append(None)

            # Create overlay plot
            unit = nodes[0].rdf[pair]["unit"]
            overlay_fig = create_overlay_plot(
                x_data,
                y_data,
                labels,
                title=f"RDF Comparison: {pair}",
                xlabel=f"r / {unit}",
                ylabel="g(r)",
                uncertainties=uncertainties,
                use_plotly=use_plotly,
            )
            figures[f"overlay_{pair}"] = overlay_fig

            # Create difference plot (if x-grids are compatible)
            try:
                diff_fig = create_difference_plot(
                    x_data,
                    y_data,
                    labels,
                    title=f"RDF Differences: {pair}",
                    xlabel=f"r / {unit}",
                    ylabel="Δg(r)",
                    reference_idx=0,
                    use_plotly=use_plotly,
                )
                figures[f"difference_{pair}"] = diff_fig
            except Exception as e:
                log.warning(f"Could not create difference plot for {pair}: {e}")

            # Extract peak statistics
            peak_positions[pair] = []
            peak_heights[pair] = []
            coordination_numbers[pair] = []

            for i, (x, y) in enumerate(zip(x_data, y_data)):
                # Find first peak (exclude r < 1.0 to avoid artifacts)
                mask = x > 1.0
                if np.any(mask):
                    x_valid = x[mask]
                    y_valid = y[mask]
                    peak_idx = np.argmax(y_valid)
                    peak_positions[pair].append(x_valid[peak_idx])
                    peak_heights[pair].append(y_valid[peak_idx])

                    # Estimate coordination number (integrate first peak)
                    # Find first minimum after peak
                    try:
                        min_idx = peak_idx + np.argmin(y_valid[peak_idx:])
                        if min_idx > peak_idx:
                            # Simple trapezoidal integration
                            r_cut = x_valid[min_idx]
                            mask_int = x <= r_cut
                            r_int = x[mask_int]
                            g_int = y[mask_int]

                            # CN = 4π * ρ * ∫ r² g(r) dr
                            density_a = nodes[i].rdf[pair]["number_density_a"]
                            density_b = nodes[i].rdf[pair]["number_density_b"]
                            # Use geometric mean of densities
                            rho = np.sqrt(density_a * density_b)

                            integrand = r_int**2 * g_int
                            cn = 4 * np.pi * rho * np.trapz(integrand, r_int)
                            coordination_numbers[pair].append(cn)
                        else:
                            coordination_numbers[pair].append(np.nan)
                    except Exception:
                        coordination_numbers[pair].append(np.nan)
                else:
                    peak_positions[pair].append(np.nan)
                    peak_heights[pair].append(np.nan)
                    coordination_numbers[pair].append(np.nan)

        # Create summary bar charts
        if peak_positions:
            peak_pos_fig = create_bar_comparison(
                peak_positions,
                None,
                labels,
                title="First Peak Positions",
                ylabel=f"r / {unit}",
                use_plotly=use_plotly,
            )
            figures["peak_positions"] = peak_pos_fig

        if peak_heights:
            peak_height_fig = create_bar_comparison(
                peak_heights,
                None,
                labels,
                title="First Peak Heights",
                ylabel="g(r)",
                use_plotly=use_plotly,
            )
            figures["peak_heights"] = peak_height_fig

        if coordination_numbers:
            cn_fig = create_bar_comparison(
                coordination_numbers,
                None,
                labels,
                title="Coordination Numbers (First Shell)",
                ylabel="CN",
                use_plotly=use_plotly,
            )
            figures["coordination_numbers"] = cn_fig

        return {"figures": figures}

    def run(self):
        """Executes the RDF calculation workflow."""
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.figures_path.mkdir(parents=True, exist_ok=True)
        self.rdf = {}

        # --- Uncertainty Setup ---
        use_uncertainty = self.uncertainties is not None
        if use_uncertainty:
            if self.temperature is None:
                raise ValueError(
                    "temperature parameter must be set when uncertainties is enabled"
                )
            log.info(
                f"Uncertainty quantification enabled using '{self.uncertainties}' "
                f"at T={self.temperature} K (CEA method)"
            )

        # --- Data Loading and Unit Setup ---
        io = self.data
        if isinstance(io, znh5md.IO):
            include_list = ["position", "box"]
            if use_uncertainty:
                include_list.append(self.uncertainties)
            io.include = include_list

        frames = io[self.start : self.stop : self.step]
        ureg = pint.UnitRegistry()
        position_unit = ureg.angstrom

        # --- Extract Energy Ensemble Data ---
        energy_ensemble = None
        if use_uncertainty:
            log.info("Extracting energy ensemble data from frames...")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
            ) as progress:
                energy_task = progress.add_task(
                    "[cyan]Extracting energies...", total=len(frames)
                )
                energy_list = []
                for frame in frames:
                    if hasattr(frame, "calc") and frame.calc is not None:
                        ensemble = frame.calc.results.get(self.uncertainties, None)
                        if ensemble is None:
                            raise ValueError(
                                f"Frame missing '{self.uncertainties}' in calc.results"
                            )
                        energy_list.append(ensemble)
                    else:
                        raise ValueError(
                            f"Frame has no calculator attached. Cannot extract '{self.uncertainties}'"
                        )
                    progress.update(energy_task, advance=1)
            energy_ensemble = jnp.array(energy_list)
            log.info(f"Extracted energy ensemble: shape {energy_ensemble.shape}")

        # --- Molecule Identification ---
        graph = rdkit2ase.ase2networkx(frames[0], suggestions=self.structures)
        molecules: dict[str, tuple[tuple[int, ...]]] = {}
        masses: dict[str, np.ndarray] = {}

        if not self.structures:
            log.info("No structures provided, grouping by element for atom-atom RDFs.")
            molecules = {
                k: tuple((i,) for i in v) for k, v in frames[0].symbols.indices().items()
            }
        else:
            log.info("Structures provided. Grouping atoms for COM-COM RDFs.")
            for structure in self.structures:
                matches = rdkit2ase.match_substructure(
                    rdkit2ase.networkx2ase(graph), smiles=structure
                )
                if not matches:
                    log.warning(f"No matches found for structure {structure}")
                    continue
                molecules[structure] = matches
                masses[structure] = np.array(frames[0].get_masses()[list(matches[0])])

        # --- Pre-computation Step: Get all particle positions (Atoms or COMs) ---
        all_positions = jnp.array([frame.get_positions() for frame in frames])
        all_cells = jnp.array([frame.get_cell().__array__() for frame in frames])
        particle_positions = {}

        if self.structures:
            log.info("Preprocessing: Calculating COMs for all frames...")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
            ) as progress:
                com_task = progress.add_task(
                    "[cyan]Computing COMs...", total=len(molecules)
                )
                for name, mol_indices_tuples in molecules.items():
                    n_molecules = len(mol_indices_tuples)
                    if n_molecules == 0:
                        progress.update(com_task, advance=1)
                        continue
                    n_atoms_per_mol = len(mol_indices_tuples[0])
                    atom_indices = jnp.array(mol_indices_tuples)
                    mol_masses = jnp.array(masses[name])

                    # Process frames sequentially with jax.lax.map
                    def process_frame(frame_args):
                        frame_positions, frame_cell = frame_args
                        return _calculate_pbc_aware_com_for_frame(
                            frame_positions,
                            frame_cell,
                            atom_indices,
                            mol_masses,
                            n_molecules,
                            n_atoms_per_mol,
                        )

                    particle_positions[name] = jax.lax.map(
                        process_frame, (all_positions, all_cells)
                    )
                    progress.update(com_task, advance=1)
        else:
            log.info("Preprocessing: Gathering atomic positions...")
            for name, atom_indices_tuples in molecules.items():
                indices = jnp.array([idx[0] for idx in atom_indices_tuples])
                particle_positions[name] = all_positions[:, indices, :]

        # --- RDF Calculation Loop with Enhanced Progress Tracking ---
        structure_names = sorted(molecules.keys())
        pairs = generate_sorted_pairs(structure_names)

        # Create Rich progress bar with two tasks
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            overall_task = progress.add_task(
                "[bold green]Overall Progress", total=len(pairs)
            )
            current_task = progress.add_task("[cyan]Current Pair", total=1, visible=False)

            for struct_a, struct_b in pairs:
                if (
                    struct_a not in particle_positions
                    or struct_b not in particle_positions
                ):
                    progress.update(overall_task, advance=1)
                    continue

                pair_key = f"{struct_a}-{struct_b}"
                safe_pair_key = sanitize_structure_name(pair_key)
                pos_a = particle_positions[struct_a]
                pos_b = particle_positions[struct_b]

                # Define bins based on the smallest box size to avoid artifacts
                min_box_dim = np.min(np.linalg.norm(all_cells, axis=2))
                r_max = min_box_dim / 2.0
                bin_edges = np.arange(0, r_max, self.bin_width)
                bin_centers = bin_edges[:-1] + self.bin_width / 2.0

                # Calculate number of batches
                n_frames = len(all_cells)
                n_batches_per_model = (n_frames + self.batch_size - 1) // self.batch_size

                # --- RDF Calculation ---
                # Calculate total batches for progress tracking
                if use_uncertainty:
                    assert energy_ensemble is not None
                    n_models = energy_ensemble.shape[1]
                    total_batches = n_models * n_batches_per_model
                else:
                    total_batches = n_batches_per_model

                # Setup progress tracking
                progress.update(
                    current_task,
                    description=f"[cyan]Pair: {pair_key}",
                    total=total_batches,
                    completed=0,
                    visible=True,
                )

                # Compute RDF (unified function handles both standard and ensemble cases)
                g_r, g_r_std, g_r_ensemble, num_dens_a, num_dens_b = compute_rdf(
                    pos_a,
                    pos_b,
                    all_cells,
                    jnp.array(bin_edges),
                    batch_size=self.batch_size,
                    exclude_self=(struct_a == struct_b),
                    energy_ensemble=energy_ensemble,
                    temperature=self.temperature,
                    progress=progress,
                    batch_task_id=current_task,
                )

                # Convert to numpy arrays
                g_r_np = np.asarray(g_r)
                g_r_std_np = np.asarray(g_r_std) if g_r_std is not None else None
                g_r_ensemble_np = (
                    np.asarray(g_r_ensemble) if g_r_ensemble is not None else None
                )

                # Hide current task after completion
                progress.update(current_task, visible=False)

                # --- Save Data and Metrics ---
                data_file = self.data_path / f"rdf_{safe_pair_key}.txt"
                if use_uncertainty:
                    assert g_r_std_np is not None
                    assert g_r_ensemble_np is not None
                    # Save with uncertainty columns
                    np.savetxt(
                        data_file,
                        np.vstack([bin_centers, g_r_np, g_r_std_np]).T,
                        header=f"r ({position_unit:~P}), g(r), g(r)_std",
                    )
                    # Save full ensemble data separately
                    ensemble_file = self.data_path / f"rdf_{safe_pair_key}_ensemble.txt"
                    np.savetxt(
                        ensemble_file,
                        g_r_ensemble_np.T,
                        header=f"Per-model g(r) for {pair_key} (columns = models)",
                    )
                else:
                    np.savetxt(
                        data_file,
                        np.vstack([bin_centers, g_r_np]).T,
                        header=f"r ({position_unit:~P}), g(r)",
                    )

                self.rdf[pair_key] = {
                    "bin_centers": bin_centers.tolist(),
                    "g_r": g_r_np.tolist(),
                    "g_r_std": g_r_std_np.tolist() if use_uncertainty else None,
                    "g_r_ensemble": g_r_ensemble_np.tolist() if use_uncertainty else None,
                    "unit": f"{position_unit:~P}",
                    "number_density_a": float(num_dens_a),
                    "number_density_b": float(num_dens_b),
                }

                # --- Plotting ---
                plt.figure(figsize=(10, 6))
                plt.plot(bin_centers, g_r_np, label=pair_key, linewidth=2)

                if use_uncertainty:
                    assert g_r_std_np is not None
                    # Plot 95% confidence interval (±1.96 * σ)
                    ci_factor = 1.96
                    plt.fill_between(
                        bin_centers,
                        g_r_np - ci_factor * g_r_std_np,
                        g_r_np + ci_factor * g_r_std_np,
                        alpha=0.3,
                        label="95% CI",
                    )

                plt.xlabel(f"r  / {position_unit:~P}")
                plt.ylabel("g(r)")
                title = f"Radial Distribution Function: {pair_key}"
                if use_uncertainty:
                    title += f" (T={self.temperature} K)"
                plt.title(title)
                plt.axhline(1.0, color="grey", linestyle="--", label="Ideal Gas g(r)=1")
                plt.grid(True, linestyle="--", alpha=0.6)
                plt.legend()
                plt.tight_layout()
                plt.savefig(self.figures_path / f"rdf_{safe_pair_key}.png", dpi=300)
                plt.close()

                # Update overall progress
                progress.update(overall_task, advance=1)
