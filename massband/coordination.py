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
from massband.utils import sanitize_structure_name

log = logging.getLogger(__name__)


class CoordinationNumber(zntrack.Node):
    """Calculate coordination numbers from RDF data.

    This node depends on RadialDistributionFunction results and calculates
    the coordination number (CN) for the first shell of each RDF pair.
    The CN is found by integrating the RDF up to its first minimum.
    For a pair of species A and B, it calculates both CN(A-B) (B around A)
    and CN(B-A) (A around B) by using the respective partial number densities
    provided by the upstream RadialDistributionFunction node.
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
        """Find the position of the first minimum in the RDF."""
        # Only consider data beyond r=1.0 Å to avoid spurious minima
        mask = r > 1.0
        r_filtered, g_r_filtered = r[mask], g_r[mask]

        if len(r_filtered) == 0:
            log.warning(
                "No data points beyond 1.0 Å, using maximum integration distance."
            )
            return self.max_integration_distance

        g_r_smooth = uniform_filter1d(g_r_filtered, size=3)
        minima_indices = argrelmin(g_r_smooth, order=2)[0]

        if len(minima_indices) == 0:
            log.warning("No minima found in RDF, using maximum integration distance.")
            return self.max_integration_distance

        for idx in minima_indices:
            if g_r_smooth[idx] < self.density_threshold:
                return min(r_filtered[idx], self.max_integration_distance)

        return min(r_filtered[minima_indices[0]], self.max_integration_distance)

    def _calculate_coordination_number(
        self,
        r: np.ndarray,
        g_r: np.ndarray,
        number_density: float,
        first_min_distance: float,
    ) -> float:
        """Calculate coordination number by integrating RDF."""
        mask = r <= first_min_distance
        r_int, g_r_int = r[mask], g_r[mask]

        if len(r_int) < 2:
            log.warning("Insufficient data points for integration.")
            return 0.0

        integrand = 4 * np.pi * r_int**2 * g_r_int
        try:
            return number_density * simpson(integrand, r_int)
        except (ValueError, IndexError):
            return number_density * np.trapezoid(integrand, r_int)

    def _plot_coordination_analysis(
        self,
        pair_name: str,
        r: np.ndarray,
        g_r: np.ndarray,
        first_min_distance: float,
        coordination_number: float,
        number_density: float,
        title_override: str = None,
    ):
        """Create a plot showing the RDF and coordination number analysis."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), constrained_layout=True)

        title = title_override or f"RDF Analysis: {pair_name}"
        fig.suptitle(title, fontsize=16)

        # Top plot: RDF
        ax1.plot(r, g_r, "b-", label="g(r)", linewidth=2)
        ax1.axvline(
            first_min_distance,
            color="r",
            ls="--",
            label=f"1st min: {first_min_distance:.2f} Å",
        )
        ax1.fill_between(
            r[r <= first_min_distance],
            g_r[r <= first_min_distance],
            color="g",
            alpha=0.3,
            label="Integration region",
        )
        ax1.axhline(
            self.density_threshold,
            color="grey",
            ls=":",
            label=f"Density threshold: {self.density_threshold}",
        )
        ax1.set_xlabel("Distance r (Å)")
        ax1.set_ylabel("g(r)")
        ax1.set_title("Radial Distribution Function")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, min(np.max(r), 15))

        # Bottom plot: Running CN
        running_cn = np.zeros_like(r)
        for i in range(1, len(r)):
            mask = r <= r[i]
            if np.sum(mask) > 1:
                integrand = 4 * np.pi * r[mask] ** 2 * g_r[mask]
                running_cn[i] = number_density * np.trapezoid(integrand, r[mask])

        ax2.plot(r, running_cn, "g-", linewidth=2, label="Running CN")
        ax2.axvline(first_min_distance, color="r", ls="--")
        ax2.axhline(
            coordination_number,
            color="b",
            ls="-",
            label=f"Final CN = {coordination_number:.2f}",
        )
        ax2.set_xlabel("Distance r (Å)")
        ax2.set_ylabel("Coordination Number")
        ax2.set_title("Running Coordination Number")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, min(np.max(r), 15))
        ax2.set_ylim(0, max(coordination_number * 2.5, 5))

        safe_pair_name = sanitize_structure_name(pair_name)
        save_path = self.figures / f"coordination_{safe_pair_name}.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def run(self):
        """Calculate coordination numbers for all RDF pairs."""
        self.figures.mkdir(parents=True, exist_ok=True)
        self.coordination_numbers = {}
        self.first_shell_distances = {}

        bin_width = self.rdf.bin_width
        partial_densities = self.rdf.partial_number_densities

        log.info(
            f"Calculating coordination numbers using partial densities: {partial_densities}"
        )

        for pair_key, g_r_list in self.rdf.results.items():
            g_r = np.array(g_r_list)
            r = np.arange(len(g_r)) * bin_width + bin_width / 2.0

            if len(g_r) == 0 or not np.all(np.isfinite(g_r)):
                log.warning(f"Skipping '{pair_key}' due to invalid RDF data.")
                continue

            try:
                species_A, species_B = pair_key.split("|")
            except ValueError:
                log.error(
                    f"Could not parse pair key '{pair_key}'. Assumes 'A|B' format. Skipping."
                )
                continue

            # Find the first minimum, which is common for both calculations (A-B and B-A)
            first_min_distance = self._find_first_minimum(r, g_r)

            # --- Perform two calculations for each pair: A-B and B-A ---
            # The order of species in the tuple determines which density is used.
            # (Center, Neighbor) -> use density of Neighbor
            calculations = [(species_A, species_B)]
            if species_A != species_B:
                calculations.append((species_B, species_A))

            for center_species, neighbor_species in calculations:
                cn_key = f"{center_species}|{neighbor_species}"
                rho_neighbor = partial_densities.get(neighbor_species)

                if rho_neighbor is None:
                    log.error(
                        f"Partial density for '{neighbor_species}' not found. Cannot calculate CN for '{cn_key}'."
                    )
                    continue

                # Calculate the coordination number
                cn = self._calculate_coordination_number(
                    r, g_r, rho_neighbor, first_min_distance
                )
                self.coordination_numbers[cn_key] = cn
                self.first_shell_distances[cn_key] = first_min_distance

                log.info(
                    f"CN for '{cn_key}' (neighbors around center): {cn:.2f} "
                    f"[shell at {first_min_distance:.2f} Å, ρ({neighbor_species})={rho_neighbor:.4f} Å⁻³]"
                )

                # Create analysis plot with a descriptive title
                plot_title = f"Coordination of {neighbor_species} around {center_species}"
                self._plot_coordination_analysis(
                    pair_name=cn_key,
                    r=r,
                    g_r=g_r,
                    first_min_distance=first_min_distance,
                    coordination_number=cn,
                    number_density=rho_neighbor,
                    title_override=plot_title,
                )

        log.info("✅ Coordination number analysis completed.")
