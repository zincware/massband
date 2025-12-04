"""Potential of Mean Force (PMF) calculation from RDF data."""

import logging
import typing as t
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pint
import zntrack

from massband.abc import ComparisonResults
from massband.comparison_utils import (
    create_bar_comparison,
    create_difference_plot,
    create_overlay_plot,
)
from massband.rdf.core import RDFData
from massband.utils import sanitize_structure_name

log = logging.getLogger(__name__)

ureg = pint.UnitRegistry()


class PMFResults(t.TypedDict):
    """TypedDict for storing PMF results."""

    r: list[float]
    pmf: list[float]
    pmf_std: list[float] | None
    unit: str


class PotentialOfMeanForce(zntrack.Node):
    """Calculate potential of mean force (PMF) from RDF data.

    The PMF is calculated using the relation:
    PMF(r) = -kT * ln(g(r))

    If RDF uncertainties are available (from ensemble calculations), the PMF
    uncertainty is propagated via: δPMF = kT * δg(r) / g(r)

    The resulting PMF is normalized such that it approaches zero at large
    distances, representing the energy of non-interacting particles.

    Parameters
    ----------
    data : dict[str, RDFData]
        Dictionary containing RDF data for each atom/molecule pair. Keys are
        pair names (e.g., "Li-Li"), values are RDFData TypedDict with fields:
        bin_centers, g_r, g_r_std, g_r_ensemble, unit, number_density_a, number_density_b.
    temperature : float, optional
        Temperature in Kelvin for the calculation, by default 300.0.
    min_gdr_threshold : float, optional
        A small value to prevent taking the logarithm of zero. This acts as
        a floor for g(r) values, by default 1e-9.

    Attributes
    ----------
    pmf : dict[str, PMFResults]
        The output dictionary storing the PMF results for each pair. Includes
        uncertainties if RDF uncertainties were available.
    figures : Path
        The output path where PMF analysis plots are saved with 95% confidence
        intervals when uncertainties are available.
    """

    data: dict[str, RDFData] = zntrack.deps()

    temperature: float = zntrack.params(300.0)
    min_gdr_threshold: float = zntrack.params(1e-9)

    # Outputs
    pmf: dict[str, PMFResults] = zntrack.outs()
    figures: Path = zntrack.outs_path(zntrack.nwd / "pmf_figures")

    @staticmethod
    def compare(  # noqa: C901
        *nodes: "PotentialOfMeanForce",
        labels: list[str] | None = None,
        pairs: list[str] | None = None,
        align_minima: bool = False,
        use_plotly: bool = False,
    ) -> ComparisonResults:
        """Compare PMF curves from multiple calculations.

        Parameters
        ----------
        *nodes : PotentialOfMeanForce
            Multiple PMF node instances to compare
        labels : list[str] | None
            Labels for each node (e.g., ["ML potential", "DFT", "Experiment"]).
            If None, uses node indices as labels.
        pairs : list[str] | None
            Specific pairs to compare (e.g., ["Li-F", "Li-Li"]).
            If None, compares all common pairs across all nodes.
        align_minima : bool
            If True, shifts PMF curves vertically to align their first minima.
        use_plotly : bool
            If True, creates plotly figures; otherwise matplotlib.

        Returns
        -------
        ComparisonResults
            Dictionary containing comparative PMF plots with keys:
            - "overlay_{pair}": overlay plot for each pair
            - "difference_{pair}": difference plot for each pair
            - "barrier_heights": bar chart comparing energy barriers
            - "well_depths": bar chart comparing well depths
            - "minima_positions": bar chart comparing position of first minimum
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
        common_pairs = set(nodes[0].pmf.keys())
        for node in nodes[1:]:
            common_pairs &= set(node.pmf.keys())

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
        figures: dict[str, plt.Figure] = {}

        # Track statistics for all pairs
        barrier_heights: dict[str, list[float]] = {}
        well_depths: dict[str, list[float]] = {}
        minima_positions: dict[str, list[float]] = {}

        # Compare each pair
        for pair in common_pairs:
            # Extract data for this pair from all nodes
            x_data = []
            y_data = []
            uncertainties = []

            for node in nodes:
                pmf_data = node.pmf[pair]
                r = np.array(pmf_data["r"])
                pmf = np.array(pmf_data["pmf"])

                # Handle alignment if requested
                if align_minima:
                    # Find first minimum (where r > 1.0)
                    finite_mask = np.isfinite(pmf) & (r > 1.0)
                    if np.any(finite_mask):
                        min_val = np.nanmin(pmf[finite_mask])
                        pmf = pmf - min_val  # Shift to align minima at 0

                x_data.append(r)
                y_data.append(pmf)

                # Get uncertainties if available
                if pmf_data.get("pmf_std") is not None:
                    uncertainties.append(np.array(pmf_data["pmf_std"]))
                else:
                    uncertainties.append(None)

            # Create overlay plot
            unit = nodes[0].pmf[pair]["unit"]
            overlay_title = f"PMF Comparison: {pair}"
            if align_minima:
                overlay_title += " (aligned at first minimum)"

            overlay_fig = create_overlay_plot(
                x_data,
                y_data,
                labels,
                title=overlay_title,
                xlabel="r / Å",
                ylabel=f"PMF / {unit}",
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
                    title=f"PMF Differences: {pair}",
                    xlabel="r / Å",
                    ylabel=f"ΔPMF / {unit}",
                    reference_idx=0,
                    use_plotly=use_plotly,
                )
                figures[f"difference_{pair}"] = diff_fig
            except Exception as e:
                log.warning(f"Could not create difference plot for {pair}: {e}")

            # Extract statistics
            barrier_heights[pair] = []
            well_depths[pair] = []
            minima_positions[pair] = []

            for r, pmf in zip(x_data, y_data):
                # Only consider finite values where r > 1.0
                finite_mask = np.isfinite(pmf) & (r > 1.0)

                if np.any(finite_mask):
                    r_valid = r[finite_mask]
                    pmf_valid = pmf[finite_mask]

                    # Find first minimum
                    min_idx = np.argmin(pmf_valid)
                    min_pos = r_valid[min_idx]
                    min_val = pmf_valid[min_idx]

                    minima_positions[pair].append(min_pos)
                    well_depths[pair].append(min_val)

                    # Find first maximum after minimum (barrier)
                    if min_idx < len(pmf_valid) - 1:
                        max_idx = min_idx + np.argmax(pmf_valid[min_idx:])
                        if max_idx > min_idx:
                            barrier = pmf_valid[max_idx] - min_val
                            barrier_heights[pair].append(barrier)
                        else:
                            barrier_heights[pair].append(np.nan)
                    else:
                        barrier_heights[pair].append(np.nan)
                else:
                    barrier_heights[pair].append(np.nan)
                    well_depths[pair].append(np.nan)
                    minima_positions[pair].append(np.nan)

        # Create summary bar charts
        if barrier_heights:
            barrier_fig = create_bar_comparison(
                barrier_heights,
                None,
                labels,
                title="Energy Barrier Heights",
                ylabel=f"Barrier / {unit}",
                use_plotly=use_plotly,
            )
            figures["barrier_heights"] = barrier_fig

        if well_depths:
            well_fig = create_bar_comparison(
                well_depths,
                None,
                labels,
                title="Well Depths",
                ylabel=f"Well Depth / {unit}",
                use_plotly=use_plotly,
            )
            figures["well_depths"] = well_fig

        if minima_positions:
            minima_fig = create_bar_comparison(
                minima_positions,
                None,
                labels,
                title="First Minimum Positions",
                ylabel="r / Å",
                use_plotly=use_plotly,
            )
            figures["minima_positions"] = minima_fig

        return {"figures": figures}

    def _calculate_pmf(
        self,
        r: np.ndarray,
        g_r: np.ndarray,
        temperature: float,
        g_r_std: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Calculate and normalize the PMF from RDF data with uncertainty propagation.

        The method first calculates the raw PMF using PMF(r) = -kT * ln(g(r)).
        If uncertainties are provided, propagates them through the logarithm:
        δPMF = kT * δg(r) / g(r)

        It then normalizes the PMF by subtracting the average value of its
        tail, ensuring that the PMF approaches zero at large distances where
        inter-particle interactions are negligible.

        Parameters
        ----------
        r : np.ndarray
            Array of distances in Angstroms.
        g_r : np.ndarray
            Array of radial distribution function values.
        temperature : float
            The system temperature in Kelvin.
        g_r_std : np.ndarray, optional
            Standard deviation of g(r) from ensemble. If provided, PMF uncertainty
            is calculated.

        Returns
        -------
        pmf_ev : np.ndarray
            The normalized Potential of Mean Force in electron-volts (eV).
        pmf_std_ev : np.ndarray or None
            Standard deviation of PMF in eV, or None if g_r_std not provided.
        """
        # Create temperature with units for calculation
        T = temperature * ureg.kelvin
        kT_ev = (ureg.boltzmann_constant * T).to("eV").magnitude

        # Apply a floor to g(r) to prevent log(0) errors
        g_r_safe = np.maximum(g_r, self.min_gdr_threshold)

        # Calculate raw PMF using Boltzmann constant from pint
        pmf_dimensionless = -np.log(g_r_safe)
        pmf_quantity = ureg.boltzmann_constant * T * pmf_dimensionless
        pmf_ev = pmf_quantity.to("eV").magnitude

        # Set PMF to NaN where original g(r) was below the threshold.
        # This is important for plotting and normalization.
        pmf_ev[g_r < self.min_gdr_threshold] = np.nan

        # Calculate PMF uncertainty if RDF uncertainty is provided
        pmf_std_ev = None
        if g_r_std is not None:
            # Propagate uncertainty: δPMF = kT * δg(r) / g(r)
            # Only compute where g(r) is above threshold
            pmf_std_ev = np.full_like(pmf_ev, np.nan)
            valid_mask = g_r >= self.min_gdr_threshold
            pmf_std_ev[valid_mask] = kT_ev * g_r_std[valid_mask] / g_r[valid_mask]

        # **Normalization**: Shift PMF to be zero at large distances
        # We define the "tail" as the last 20% of the distance range
        tail_start_idx = int(len(r) * 0.8)
        if tail_start_idx < len(r):
            tail_pmf = pmf_ev[tail_start_idx:]
            finite_tail_pmf = tail_pmf[np.isfinite(tail_pmf)]
            if finite_tail_pmf.size > 0:
                offset = np.mean(finite_tail_pmf)
                pmf_ev -= offset
                log.debug(f"Applied PMF offset of {offset:.4f} eV.")
            else:
                log.warning("Could not normalize PMF as tail contains no finite values.")
        else:
            log.warning("Not enough data points to normalize PMF tail.")

        # Note: pmf_std_ev is not affected by the offset (constant shift)

        return pmf_ev, pmf_std_ev

    def _plot_pmf_analysis(
        self,
        pair_name: str,
        r: np.ndarray,
        g_r: np.ndarray,
        pmf: np.ndarray,
        g_r_std: np.ndarray | None = None,
        pmf_std: np.ndarray | None = None,
        title_override: str = None,
    ):
        """Create and save a two-panel plot of RDF and PMF with uncertainty bands.

        This revised version highlights regions where the PMF is artificially
        cut because g(r) approaches zero, indicating a very high energy barrier.
        If uncertainties are provided, plots confidence intervals.

        Parameters
        ----------
        pair_name : str
            The name of the atom pair (e.g., "O-H").
        r : np.ndarray
            Array of distances in Angstroms.
        g_r : np.ndarray
            Array of radial distribution function values.
        pmf : np.ndarray
            Array of potential of mean force values in eV.
        g_r_std : np.ndarray, optional
            Standard deviation of g(r). If provided, plots 95% CI on RDF.
        pmf_std : np.ndarray, optional
            Standard deviation of PMF. If provided, plots 95% CI on PMF.
        title_override : str, optional
            A custom title for the entire figure, by default None.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), constrained_layout=True)

        title = title_override or f"PMF Analysis: {pair_name}"
        fig.suptitle(title, fontsize=16)

        # Top panel: RDF
        ax1.plot(r, g_r, "b-", label="g(r)", linewidth=2)

        # Add 95% confidence interval if uncertainty provided
        if g_r_std is not None:
            ci_factor = 1.96
            ax1.fill_between(
                r,
                g_r - ci_factor * g_r_std,
                g_r + ci_factor * g_r_std,
                alpha=0.3,
                color="blue",
                label="95% CI",
            )

        ax1.axhline(1.0, color="grey", ls=":", alpha=0.7, label="g(r) = 1 (No structure)")
        ax1.set_xlabel("Distance r / Å")
        ax1.set_ylabel("g(r)")
        ax1.set_title("Radial Distribution Function")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(left=0, right=min(np.max(r, initial=15), 15))
        ax1.set_ylim(bottom=0)

        # Bottom panel: PMF
        finite_mask = np.isfinite(pmf)
        if np.any(finite_mask):
            # --- Plot in contiguous segments to avoid misleading lines ---
            # Find indices of all finite PMF values
            finite_indices = np.where(finite_mask)[0]
            # Find where the breaks are in the indices (a jump > 1)
            breaks = np.where(np.diff(finite_indices) != 1)[0] + 1
            # Split the indices into segments at the break points
            segments = np.split(finite_indices, breaks)

            # Plot each segment as a separate line
            for segment_indices in segments:
                if segment_indices.size > 1:  # Need at least 2 points to draw a line
                    ax2.plot(
                        r[segment_indices],
                        pmf[segment_indices],
                        "r-",
                        linewidth=2,
                        label="PMF",
                    )

            # Add 95% confidence interval if uncertainty provided
            if pmf_std is not None:
                ci_factor = 1.96
                # Only plot CI where both PMF and PMF_std are finite
                valid_ci_mask = np.isfinite(pmf) & np.isfinite(pmf_std)
                if np.any(valid_ci_mask):
                    ax2.fill_between(
                        r,
                        pmf - ci_factor * pmf_std,
                        pmf + ci_factor * pmf_std,
                        alpha=0.3,
                        color="red",
                        label="95% CI",
                    )

            ax2.axhline(0.0, color="grey", ls=":", alpha=0.7, label="PMF = 0 (Reference)")
        else:
            ax2.text(
                0.5,
                0.5,
                "No finite PMF values to plot",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )

        # Set reasonable y-limits for visualization before adding shaded region
        finite_pmf = pmf[finite_mask]
        if finite_pmf.size > 0:
            pmf_min, pmf_max = np.min(finite_pmf), np.max(finite_pmf)
            # Add padding but clamp extreme values for a clear plot
            y_min_lim = max(pmf_min - 0.1, -5.0)
            y_max_lim = min(pmf_max + 0.1, 5.0)
            ax2.set_ylim(y_min_lim, y_max_lim)

        # Highlight regions where PMF is cut off
        cutoff_mask = g_r < self.min_gdr_threshold
        if np.any(cutoff_mask):
            ax2.fill_between(
                r,
                *ax2.get_ylim(),
                where=cutoff_mask,
                facecolor="lightgray",  # Changed from orange
                alpha=0.5,
                label="PMF → ∞ (g(r) ≈ 0)",
                interpolate=True,
            )
            log.info(f"Highlighted cut-off PMF regions for '{pair_name}'.")

        ax2.set_xlabel("Distance r / Å")
        ax2.set_ylabel("PMF / eV")
        ax2.set_title(f"Potential of Mean Force (T = {self.temperature} K)")

        # Ensure legend is not cluttered with duplicate entries
        handles, labels = ax2.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax2.legend(by_label.values(), by_label.keys())

        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(left=0, right=min(np.max(r, initial=15), 15))

        safe_pair_name = sanitize_structure_name(pair_name)
        save_path = self.figures / f"pmf_{safe_pair_name}.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def run(self):
        """Execute the PMF calculation for all RDF pairs.

        This method iterates through each atom pair in the input RDF data,
        calculates the corresponding PMF, stores the results in the `pmf`
        attribute, and generates analysis plots.
        """
        self.figures.mkdir(parents=True, exist_ok=True)
        self.pmf = {}

        log.info(f"Calculating PMF at T = {self.temperature} K")

        for pair_key, rdf_data in self.data.items():
            # Extract distances from bin centers
            r = np.array(rdf_data["bin_centers"])
            g_r = np.array(rdf_data["g_r"])

            if g_r.size == 0 or not np.any(np.isfinite(g_r)):
                log.warning(f"Skipping '{pair_key}' due to empty or invalid RDF data.")
                continue

            # Extract uncertainty if available
            g_r_std = None
            if rdf_data.get("g_r_std") is not None:
                g_r_std = np.array(rdf_data["g_r_std"])
                log.info(f"Using RDF uncertainties for '{pair_key}' PMF calculation.")

            # Calculate the normalized PMF with uncertainty propagation
            pmf_values, pmf_std = self._calculate_pmf(
                r, g_r, self.temperature, g_r_std=g_r_std
            )

            # Store results correctly using the PMFResults TypedDict structure
            self.pmf[pair_key] = {
                "r": r.tolist(),
                "pmf": pmf_values.tolist(),  # NaNs will be serialized as null
                "pmf_std": pmf_std.tolist() if pmf_std is not None else None,
                "unit": "eV",
            }

            # Log summary statistics
            finite_count = np.sum(np.isfinite(pmf_values))
            log_msg = f"PMF for '{pair_key}': {finite_count}/{len(pmf_values)} finite values calculated"
            if pmf_std is not None:
                finite_std_count = np.sum(np.isfinite(pmf_std))
                log_msg += f" ({finite_std_count} with uncertainties)"
            log.info(log_msg + ".")

            # Create and save the analysis plot
            self._plot_pmf_analysis(
                pair_name=pair_key,
                r=r,
                g_r=g_r,
                pmf=pmf_values,
                g_r_std=g_r_std,
                pmf_std=pmf_std,
                title_override=f"PMF Analysis: {pair_key.replace('|', '-')}",
            )

        log.info("✅ PMF analysis completed.")
