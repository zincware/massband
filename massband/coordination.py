"""Coordination number analysis based on RDF data."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zntrack
from scipy.integrate import simpson
from scipy.ndimage import uniform_filter1d
from scipy.signal import argrelmin

from massband.rdf import RadialDistributionFunction

log = logging.getLogger(__name__)


class CoordinationNumber(zntrack.Node):
    """Calculate coordination numbers from RDF data.

    This node depends on RadialDistributionFunction results and calculates
    the coordination number for the first shell of each RDF pair by integrating
    the RDF up to the first minimum. The number density is automatically
    calculated from the ASE atoms object's cell volume and atom count.

    Parameters
    ----------
    density_threshold : float, default=0.1
        Threshold for finding first minimum in RDF (dimensionless).
    max_integration_distance : float, default=10.0
        Maximum distance to consider for integration (Å).
    """

    rdf: RadialDistributionFunction = zntrack.deps()

    # Parameters for coordination number calculation
    density_threshold: float = zntrack.params(0.1)  # Threshold for finding first minimum
    max_integration_distance: float = zntrack.params(
        10.0
    )  # Maximum distance to consider (Å)

    # Outputs
    coordination_numbers: dict[str, float] = zntrack.metrics()
    first_shell_distances: dict[str, float] = zntrack.metrics()
    figures: Path = zntrack.outs_path(zntrack.nwd / "coordination_figures")

    def _find_first_minimum(self, r: np.ndarray, g_r: np.ndarray) -> float:
        """Find the position of the first minimum in the RDF.

        Parameters
        ----------
        r : np.ndarray
            Distance array.
        g_r : np.ndarray
            RDF values.

        Returns
        -------
        float
            Distance of the first minimum.
        """
        # Only consider data beyond r=1.0 Å to avoid spurious minima at very short distances
        mask = r > 1.0
        r_filtered = r[mask]
        g_r_filtered = g_r[mask]

        if len(r_filtered) == 0:
            log.warning("No data points beyond 1.0 Å, using maximum integration distance")
            return self.max_integration_distance

        # Find local minima
        # Smooth the data slightly to avoid noise-induced minima
        g_r_smooth = uniform_filter1d(g_r_filtered, size=3)

        minima_indices = argrelmin(g_r_smooth, order=2)[0]

        if len(minima_indices) == 0:
            log.warning("No minima found, using maximum integration distance")
            return self.max_integration_distance

        # Find the first minimum that is below the density threshold
        for idx in minima_indices:
            if g_r_smooth[idx] < self.density_threshold:
                first_min_distance = r_filtered[idx]
                # Ensure we don't integrate beyond our maximum distance
                return min(first_min_distance, self.max_integration_distance)

        # If no minimum below threshold found, use the first minimum
        first_min_distance = r_filtered[minima_indices[0]]
        return min(first_min_distance, self.max_integration_distance)

    def _calculate_coordination_number(
        self,
        r: np.ndarray,
        g_r: np.ndarray,
        number_density: float,
        first_min_distance: float,
    ) -> float:
        """Calculate coordination number by integrating RDF.

        Parameters
        ----------
        r : np.ndarray
            Distance array.
        g_r : np.ndarray
            RDF values.
        number_density : float
            Number density of the system.
        first_min_distance : float
            Distance to integrate up to.

        Returns
        -------
        float
            Coordination number.
        """
        # Create mask for integration range
        mask = r <= first_min_distance
        r_int = r[mask]
        g_r_int = g_r[mask]

        if len(r_int) < 2:
            log.warning("Insufficient data points for integration")
            return 0.0

        # Calculate coordination number: CN = 4π * ρ * ∫[0 to r_min] r² * g(r) dr
        integrand = 4 * np.pi * r_int**2 * g_r_int

        # Use Simpson's rule for integration
        try:
            coordination_number = number_density * simpson(integrand, r_int)
        except ValueError:
            # Fall back to trapezoidal rule if Simpson's rule fails
            coordination_number = number_density * np.trapezoid(integrand, r_int)

        return coordination_number

    def _plot_coordination_analysis(
        self,
        pair_name: str,
        r: np.ndarray,
        g_r: np.ndarray,
        first_min_distance: float,
        coordination_number: float,
        number_density: float,
    ):
        """Create a plot showing the RDF and coordination number analysis.

        Parameters
        ----------
        pair_name : str
            Name of the atomic/molecular pair.
        r : np.ndarray
            Distance array.
        g_r : np.ndarray
            RDF values.
        first_min_distance : float
            Distance of first minimum.
        coordination_number : float
            Calculated coordination number.
        number_density : float
            Number density used for coordination number calculation.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

        # Top plot: RDF with integration range
        ax1.plot(r, g_r, "b-", label="g(r)", linewidth=2)
        ax1.axvline(
            first_min_distance,
            color="red",
            linestyle="--",
            label=f"1st minimum: {first_min_distance:.2f} Å",
        )
        ax1.fill_between(
            r[r <= first_min_distance],
            g_r[r <= first_min_distance],
            alpha=0.3,
            color="green",
            label="Integration region",
        )
        ax1.axhline(
            self.density_threshold,
            color="gray",
            linestyle=":",
            label=f"Density threshold: {self.density_threshold}",
        )

        ax1.set_xlabel("Distance r (Å)")
        ax1.set_ylabel("g(r)")
        ax1.set_title(f"RDF: {pair_name}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, min(np.max(r), 15))

        # Bottom plot: Running coordination number
        running_cn = np.zeros_like(r)
        # Use the same number density as the fitted coordination number
        estimated_density = number_density

        for i in range(1, len(r)):
            mask = r <= r[i]
            r_temp = r[mask]
            g_r_temp = g_r[mask]
            if len(r_temp) > 1:
                integrand = 4 * np.pi * r_temp**2 * g_r_temp
                running_cn[i] = estimated_density * np.trapezoid(integrand, r_temp)

        ax2.plot(r, running_cn, "g-", linewidth=2, label="Running CN")
        ax2.axvline(first_min_distance, color="red", linestyle="--")
        ax2.axhline(
            coordination_number,
            color="blue",
            linestyle="-",
            label=f"CN = {coordination_number:.2f}",
        )

        ax2.set_xlabel("Distance r (Å)")
        ax2.set_ylabel("Coordination Number")
        ax2.set_title(f"Running Coordination Number: {pair_name}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, min(np.max(r), 15))
        # Limit y-axis to 4x the fitted coordination number for better visibility
        ax2.set_ylim(0, max(coordination_number * 4, 5))

        plt.tight_layout()

        # Save the plot
        safe_pair_name = pair_name.replace("|", "_").replace(" ", "_")
        save_path = self.figures / f"coordination_{safe_pair_name}.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def run(self):
        """Calculate coordination numbers for all RDF pairs."""
        self.figures.mkdir(parents=True, exist_ok=True)

        self.coordination_numbers = {}
        self.first_shell_distances = {}

        # Get the bin width from the RDF node
        bin_width = self.rdf.bin_width

        log.info(
            f"Calculating coordination numbers for {len(self.rdf.results)} RDF pairs"
        )

        for pair_key, g_r_list in self.rdf.results.items():
            # Convert to numpy array
            g_r = np.array(g_r_list)

            # Create distance array (bin centers)
            r = np.arange(len(g_r)) * bin_width + bin_width / 2.0

            # Skip if data is empty or has issues
            if len(g_r) == 0 or not np.all(np.isfinite(g_r)):
                log.warning(f"Skipping {pair_key} due to invalid data")
                continue

            # Find first minimum
            first_min_distance = self._find_first_minimum(r, g_r)
            self.first_shell_distances[pair_key] = first_min_distance

            # Get number density from RDF calculation (computed from ASE atoms object)
            estimated_number_density = self.rdf.number_density

            # Calculate coordination number
            coordination_number = self._calculate_coordination_number(
                r, g_r, estimated_number_density, first_min_distance
            )

            self.coordination_numbers[pair_key] = coordination_number

            # Create analysis plot
            self._plot_coordination_analysis(
                pair_key,
                r,
                g_r,
                first_min_distance,
                coordination_number,
                estimated_number_density,
            )

            log.info(
                f"{pair_key}: CN = {coordination_number:.2f}, "
                f"1st shell = {first_min_distance:.2f} Å"
            )

        log.info("Coordination number analysis completed")
