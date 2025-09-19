"""Potential of Mean Force (PMF) calculation from RDF data."""

import logging
import typing as t
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pint
import zntrack

from massband.rdf import RadialDistributionFunction

log = logging.getLogger(__name__)

ureg = pint.UnitRegistry()


class PMFResults(t.TypedDict):
    """TypedDict for storing PMF results."""

    r: list[float]
    pmf: list[float]
    unit: str


class PotentialOfMeanForce(zntrack.Node):
    """Calculate potential of mean force (PMF) from RDF data.

    The PMF is calculated using the relation:
    PMF(r) = -kT * ln(g(r))

    The resulting PMF is normalized such that it approaches zero at large
    distances, representing the energy of non-interacting particles.

    Attributes
    ----------
    rdf : RadialDistributionFunction
        The dependency node containing the RDF data.
    temperature : float, optional
        Temperature in Kelvin for the calculation, by default 300.0.
    min_gdr_threshold : float, optional
        A small value to prevent taking the logarithm of zero. This acts as
        a floor for g(r) values, by default 1e-9.
    pmf : dict[str, PMFResults]
        The output dictionary storing the PMF results for each pair.
    figures : Path
        The output path where PMF analysis plots are saved.
    """

    rdf: RadialDistributionFunction = zntrack.deps()

    temperature: float = zntrack.params(300.0)
    min_gdr_threshold: float = zntrack.params(1e-9)

    # Outputs
    pmf: dict[str, PMFResults] = zntrack.outs()
    figures: Path = zntrack.outs_path(zntrack.nwd / "pmf_figures")

    def _calculate_pmf(
        self, r: np.ndarray, g_r: np.ndarray, temperature: float
    ) -> np.ndarray:
        """Calculate and normalize the PMF from RDF data.

        The method first calculates the raw PMF using PMF(r) = -kT * ln(g(r)).
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

        Returns
        -------
        np.ndarray
            The normalized Potential of Mean Force in electron-volts (eV).
        """
        # Create temperature with units for calculation
        T = temperature * ureg.kelvin

        # Apply a floor to g(r) to prevent log(0) errors
        g_r_safe = np.maximum(g_r, self.min_gdr_threshold)

        # Calculate raw PMF using Boltzmann constant from pint
        pmf_dimensionless = -np.log(g_r_safe)
        pmf_quantity = ureg.boltzmann_constant * T * pmf_dimensionless
        pmf_ev = pmf_quantity.to("eV").magnitude

        # Set PMF to NaN where original g(r) was below the threshold.
        # This is important for plotting and normalization.
        pmf_ev[g_r < self.min_gdr_threshold] = np.nan

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

        return pmf_ev

    def _plot_pmf_analysis(
        self,
        pair_name: str,
        r: np.ndarray,
        g_r: np.ndarray,
        pmf: np.ndarray,
        title_override: str = None,
    ):
        """Create and save a two-panel plot of RDF and PMF.

        This revised version highlights regions where the PMF is artificially
        cut because g(r) approaches zero, indicating a very high energy barrier.

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
        title_override : str, optional
            A custom title for the entire figure, by default None.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), constrained_layout=True)

        title = title_override or f"PMF Analysis: {pair_name}"
        fig.suptitle(title, fontsize=16)

        # Top panel: RDF
        ax1.plot(r, g_r, "b-", label="g(r)", linewidth=2)
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
            # --- NEW: Plot in contiguous segments to avoid misleading lines ---
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

        safe_pair_name = pair_name.replace("|", "_").replace("-", "_").replace(" ", "_")
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

        bin_width = self.rdf.bin_width

        log.info(f"Calculating PMF at T = {self.temperature} K")

        for pair_key, rdf_data in self.rdf.rdf.items():
            g_r = np.array(rdf_data["g_r"])
            r = np.arange(len(g_r)) * bin_width + bin_width / 2.0

            if g_r.size == 0 or not np.any(np.isfinite(g_r)):
                log.warning(f"Skipping '{pair_key}' due to empty or invalid RDF data.")
                continue

            # Calculate the normalized PMF
            pmf_values = self._calculate_pmf(r, g_r, self.temperature)

            # Store results correctly using the PMFResults TypedDict structure
            self.pmf[pair_key] = {
                "r": r.tolist(),
                "pmf": pmf_values.tolist(),  # NaNs will be serialized as null
                "unit": "eV",
            }

            # Log summary statistics
            finite_count = np.sum(np.isfinite(pmf_values))
            log.info(
                f"PMF for '{pair_key}': {finite_count}/{len(pmf_values)} finite values calculated."
            )

            # Create and save the analysis plot
            self._plot_pmf_analysis(
                pair_name=pair_key,
                r=r,
                g_r=g_r,
                pmf=pmf_values,
                title_override=f"PMF Analysis: {pair_key.replace('|', '-')}",
            )

        log.info("✅ PMF analysis completed.")
