import itertools
import logging
from pathlib import Path

import jax.numpy as jnp
import plotly.graph_objects as go
import zntrack
from ase.data import atomic_numbers
from jax import vmap, jit
from tqdm import tqdm
from functools import partial

from massband.abc import ComparisonResults
from massband.dataloader import TimeBatchedLoader
from massband.rdf_fit import FIT_METHODS
from massband.rdf_plot import plot_rdf

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
    dists = mic_distances(
        pos_a, pos_b, frame_cell, exclude_self=exclude_self
    )
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


class RadialDistributionFunction(zntrack.Node):
    """
    Class to represent a radial distribution function (RDF) in a molecular dynamics simulation.
    """

    file: str | Path = zntrack.deps_path()
    batch_size: int = zntrack.params()  # You can set a default or make it configurable
    bin_width: float = zntrack.params(0.05)  # Width of the bins for RDF
    structures: list[str] | None = zntrack.params()
    start: int = zntrack.params(0)  # Starting frame index
    stop: int | None = zntrack.params(None)  # Ending frame index (exclusive)
    step: int = zntrack.params(1)  # Step size for frame selection

    bayesian: bool = zntrack.params(False)  # Whether to use Bayesian fitting
    fit_method: FIT_METHODS = zntrack.params("none")  # Method for fitting the

    results: dict[str, list[float]] = zntrack.outs()

    figures: Path = zntrack.outs_path(zntrack.nwd / "figures")

    def run(self):
        loader = TimeBatchedLoader(
            file=self.file,
            batch_size=self.batch_size,
            structures=self.structures,
            wrap=True,
            fixed_cell=True,
            com=self.structures is not None,  # Compute center of mass if structures are provided
            start=self.start,
            stop=self.stop,
            step=self.step,
        )
        
        # Initialize variables for RDF computation
        cells = None
        structure_names = None
        rdf_accumulators = {}
        total_frames = 0
        
        # Determine max distance and bin edges from first batch
        cells = loader.first_frame_cell
        # structure_names = list(first_batch_data.keys())
        structure_names = list(loader.indices.keys())

        # Compute RDF parameters from cell dimensions
        max_distance = 0.5 * float(jnp.min(jnp.linalg.norm(cells, axis=-1)))
        
        # Add safety checks for max_distance
        if not jnp.isfinite(max_distance) or max_distance <= 0:
            raise ValueError(f"Invalid max_distance: {max_distance}. Check cell dimensions.")
            
        # Limit max_distance to a reasonable value to prevent too many bins
        max_distance = min(max_distance, 50.0)  # Cap at 50 Å
        
        log.info(f"Using max_distance = {max_distance:.2f} Å with bin_width = {self.bin_width}")

        bin_edges = jnp.arange(0.0, max_distance + self.bin_width, self.bin_width)
        
        # Check if we have too many bins
        if len(bin_edges) > 10000:
            raise ValueError(f"Too many bins ({len(bin_edges)}). Consider increasing bin_width or reducing max_distance.")

        # Determine structure pairs for RDF computation
        def is_element(name):
            return name in atomic_numbers

        def sort_key(pair):
            a, b = pair
            a_is_elem = is_element(a)
            b_is_elem = is_element(b)
            if a_is_elem and b_is_elem:
                return (0, min(atomic_numbers[a], atomic_numbers[b]), max(atomic_numbers[a], atomic_numbers[b]))
            elif a_is_elem:
                return (1, atomic_numbers[a], b)
            elif b_is_elem:
                return (1, atomic_numbers[b], a)
            else:
                return (2, min(a, b), max(a, b))

        pairs = list(itertools.combinations_with_replacement(structure_names, 2))
        pairs = [tuple(sorted(pair, key=lambda x: (not is_element(x), atomic_numbers.get(x, float("inf")), x))) for pair in pairs]
        pairs = sorted(set(pairs), key=sort_key)
        
        # Initialize accumulators for each pair
        for pair in pairs:
            rdf_accumulators[pair] = jnp.zeros(len(bin_edges) - 1)
        
        # Process batches incrementally
        batch_number = 0
        pbar = tqdm(loader, desc="Processing RDF batches", unit="batch")
        
        for batch_data, batch_cells, _ in pbar:
            batch_number += 1
            batch_frames = list(batch_data.values())[0].shape[0]
            total_frames += batch_frames
            
            # Update progress bar description with current batch info
            pbar.set_description(f"Processing batch {batch_number}/{len(loader)}")
            pbar.set_postfix(frames=f"{total_frames}")
            
            # Create cells array for this batch
            batch_cells_array = jnp.tile(batch_cells[None, :, :], (batch_frames, 1, 1))
            
            for struct_a, struct_b in pairs:
                pos_a = batch_data[struct_a]
                pos_b = batch_data[struct_b]
                
                # Positions are already wrapped by TimeBatchedLoader
                g_r = compute_rdf(
                    positions_a=pos_a,
                    positions_b=pos_b,
                    cell=batch_cells_array,
                    bin_edges=bin_edges,
                    exclude_self=(struct_a == struct_b),
                    batch_size=50
                )
                pbar.set_postfix(
                    pair=f"{struct_a}-{struct_b}",
                )
                
                # Accumulate histogram weighted by number of frames
                rdf_accumulators[struct_a, struct_b] += g_r * batch_frames
        
        # Average the accumulated RDFs
        self.results = {}
        for pair, accumulated_rdf in rdf_accumulators.items():
            self.results[pair] = accumulated_rdf / total_frames
        self.figures.mkdir(exist_ok=True, parents=True)

        log.info("Creating RDF plots")
        plot_rdf(
            self.results,
            self.figures / "rdf.png",
            bin_width=self.bin_width,
            bayesian=self.bayesian,
            fit_method=self.fit_method,
        )

        log.info("Saving RDF results")
        self.results = {"|".join(k): v.tolist() for k, v in self.results.items()}

    @classmethod
    def compare(cls, *nodes: "RadialDistributionFunction") -> ComparisonResults:
        """
        Compare the Radial Distribution Functions from multiple runs by plotting
        them on top of each other.
        """
        # 1. Find common RDF pairs across all nodes to be compared.
        # The keys are strings like "struct_a|struct_b".
        all_rdf_keys = [set(node.results.keys()) for node in nodes if node.results]
        if not all_rdf_keys:
            return {"frames": [], "figures": {}}  # Return empty if no data

        common_keys = set.intersection(*all_rdf_keys)

        figures = {}
        # 2. For each common pair, create a comparison figure.
        for key in common_keys:
            fig = go.Figure()

            # 3. Add a scatter trace to the figure for each node's RDF.
            for node in nodes:
                if not node.results or key not in node.results:
                    continue

                g_r = jnp.array(node.results[key])
                # Recreate the r-values (bin centers) for the x-axis.
                # This assumes bin_width and range are consistent across compared nodes.
                bin_width = node.bin_width
                num_bins = len(g_r)
                r_values = jnp.arange(num_bins) * bin_width + bin_width / 2.0

                fig.add_trace(
                    go.Scatter(
                        x=r_values,
                        y=g_r,
                        mode="lines",
                        name=node.name,  # zntrack provides a 'name' for each node
                    )
                )

            # 4. Style the figure for clarity and aesthetics.
            pair_name = key.replace("|", "-")
            fig.update_layout(
                title_text=f"RDF Comparison for: {pair_name}",
                xaxis_title_text="r (Å)",
                yaxis_title_text="g(r)",
                legend_title_text="Compared Runs",
            )

            # 5. Add the generated figure to the results dictionary.
            figures[f"rdf_comparison_{pair_name}"] = fig

        return {"frames": [], "figures": figures}
