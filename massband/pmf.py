"""Potential of Mean Force (PMF) calculation from RDF data."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pint
import zntrack

from massband.rdf import RadialDistributionFunction

log = logging.getLogger(__name__)

ureg = pint.UnitRegistry()


class PotentialOfMeanForce(zntrack.Node):
    """Calculate potential of mean force (PMF) from RDF data.

    The PMF is calculated using the relation:
    PMF(r) = -kT * ln(g(r))
    
    where:
    - k is Boltzmann constant
    - T is temperature
    - g(r) is the radial distribution function
    
    The PMF represents the effective potential between particles as a function
    of their separation distance, providing insight into the energy landscape
    of interactions.
    """

    rdf: RadialDistributionFunction = zntrack.deps()

    # Parameters for PMF calculation
    temperature: float = zntrack.params(300.0)  # Temperature in Kelvin
    min_gdr_threshold: float = zntrack.params(1e-6)  # Minimum g(r) value to avoid ln(0)
    
    # Outputs
    pmf_values: dict[str, list[float]] = zntrack.metrics()
    figures: Path = zntrack.outs_path(zntrack.nwd / "pmf_figures")

    def _calculate_pmf(self, g_r: np.ndarray, temperature: float) -> np.ndarray:
        """Calculate PMF from RDF using PMF = -kT * ln(g(r)) in eV."""
        # Create temperature with units
        T = temperature * ureg.kelvin
        
        # Apply minimum threshold to avoid log(0)
        g_r_safe = np.maximum(g_r, self.min_gdr_threshold)
        
        # Calculate PMF using Boltzmann constant from pint
        pmf_dimensionless = -np.log(g_r_safe)
        pmf_quantity = ureg.boltzmann_constant * T * pmf_dimensionless
        
        # Convert to eV and get magnitude
        pmf_ev = pmf_quantity.to('eV').magnitude
        
        # Set PMF to NaN where original g(r) was effectively zero
        pmf_ev[g_r < self.min_gdr_threshold] = np.nan
        
        return pmf_ev

    def _plot_pmf_analysis(
        self,
        pair_name: str,
        r: np.ndarray,
        g_r: np.ndarray,
        pmf: np.ndarray,
        title_override: str = None,
    ):
        """Create a plot showing both RDF and PMF analysis."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), constrained_layout=True)

        title = title_override or f"PMF Analysis: {pair_name}"
        fig.suptitle(title, fontsize=16)

        # Top plot: RDF
        ax1.plot(r, g_r, "b-", label="g(r)", linewidth=2)
        ax1.axhline(1.0, color="grey", ls=":", alpha=0.7, label="g(r) = 1")
        ax1.set_xlabel("Distance r (Å)")
        ax1.set_ylabel("g(r)")
        ax1.set_title("Radial Distribution Function")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, min(np.max(r), 15))
        ax1.set_ylim(0, None)

        # Bottom plot: PMF
        # Only plot PMF where it's finite
        finite_mask = np.isfinite(pmf)
        if np.any(finite_mask):
            ax2.plot(r[finite_mask], pmf[finite_mask], "r-", linewidth=2, label="PMF")
            ax2.axhline(0.0, color="grey", ls=":", alpha=0.7, label="PMF = 0")
            
            # Set reasonable y-limits for PMF plot
            finite_pmf = pmf[finite_mask]
            if len(finite_pmf) > 0:
                pmf_min, pmf_max = np.nanmin(finite_pmf), np.nanmax(finite_pmf)
                # Limit extreme values for better visualization
                y_min = max(pmf_min, -10.0)
                y_max = min(pmf_max, 10.0)
                ax2.set_ylim(y_min * 1.2, y_max * 1.2)
        else:
            ax2.text(0.5, 0.5, "No finite PMF values to plot", 
                    ha='center', va='center', transform=ax2.transAxes)

        ax2.set_xlabel("Distance r / Å")
        ax2.set_ylabel("PMF / eV")
        ax2.set_title(f"Potential of Mean Force (T = {self.temperature} K)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, min(np.max(r), 15))

        safe_pair_name = pair_name.replace("|", "_").replace("-", "_").replace(" ", "_")
        save_path = self.figures / f"pmf_{safe_pair_name}.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def run(self):
        """Calculate PMF for all RDF pairs."""
        self.figures.mkdir(parents=True, exist_ok=True)
        self.pmf_values = {}

        bin_width = self.rdf.bin_width

        log.info(f"Calculating PMF at temperature {self.temperature} K")
        
        for pair_key, g_r_list in self.rdf.results.items():
            g_r = np.array(g_r_list)
            r = np.arange(len(g_r)) * bin_width + bin_width / 2.0

            if len(g_r) == 0 or not np.all(np.isfinite(g_r)):
                log.warning(f"Skipping '{pair_key}' due to invalid RDF data.")
                continue

            # Calculate PMF
            pmf = self._calculate_pmf(g_r, self.temperature)
            self.pmf_values[pair_key] = pmf.tolist()

            # Count finite PMF values for logging
            finite_count = np.sum(np.isfinite(pmf))
            total_count = len(pmf)
            
            log.info(
                f"PMF for '{pair_key}': {finite_count}/{total_count} finite values"
            )

            # Create analysis plot
            plot_title = f"PMF Analysis: {pair_key.replace('|', '-')}"
            self._plot_pmf_analysis(
                pair_name=pair_key,
                r=r,
                g_r=g_r,
                pmf=pmf,
                title_override=plot_title,
            )

        log.info("✅ PMF analysis completed.")