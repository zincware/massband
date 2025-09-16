from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipp as sc
import zntrack
from kinisi.arrhenius import Arrhenius

from massband.conductivity.einstein_helfand import KinisiEinsteinHelfandIonicConductivity
from massband.conductivity.ne import NernstEinsteinIonicConductivity


class KinisiConductivityArrhenius(zntrack.Node):
    """Perform Arrhenius analysis on ionic conductivity coefficients from multiple temperatures.

    Analyzes temperature-dependent ionic conductivity coefficients to extract activation energies
    and pre-exponential factors using Bayesian inference via the kinisi Arrhenius analyzer.

    Parameters
    ----------
    conductivity : list[KinisiEinsteinHelfandIonicConductivity]
        List of conductivity analysis nodes at different temperatures.
    temperatures : list[float]
        Corresponding temperatures in Kelvin for each conductivity analysis.
    reference : str | Path | None, default=None
        Path to CSV file containing reference conductivity data. First row should contain
        'temperature' followed by analysis names, subsequent rows contain temperature
        values and corresponding conductivity coefficients.
    reference_units : str, default="S/m"
        Units of the reference conductivity coefficients for unit conversion.

    Attributes
    ----------
    figures_path : Path
        Output directory for Arrhenius plots.

    Examples
    --------
    >>> with project:
    ...     arrhenius = massband.KinisiConductivityArrhenius(
    ...         conductivity=[cond_300K, cond_350K, cond_400K],
    ...     )
    >>> project.repro()
    """

    conductivity: list[
        KinisiEinsteinHelfandIonicConductivity | NernstEinsteinIonicConductivity
    ] = zntrack.deps()
    figures_path: Path = zntrack.outs_path(zntrack.nwd / "figures")
    reference: str | Path | None = zntrack.deps_path()
    reference_units: str = zntrack.params("S/m")

    def run(self):
        self.figures_path.mkdir(parents=True, exist_ok=True)
        temperatures = [x.temperature for x in self.conductivity]

        # Extract conductivity data across all temperatures
        sigma = {
            "mean": [x.conductivity["mean"] for x in self.conductivity],
            "var": [x.conductivity["var"] for x in self.conductivity],
        }

        td = sc.DataArray(
            data=sc.array(
                dims=["temperature"],
                values=sigma["mean"],
                variances=sigma["var"],
                unit=sc.Unit("S/m"),
            ),
            coords={
                "temperature": sc.Variable(
                    dims=["temperature"], values=temperatures, unit="K"
                )
            },
        )

        # Perform Arrhenius analysis
        s = Arrhenius(td)

        # Create Arrhenius plot
        self._plot_arrhenius(td, s)

    def _load_reference_data(self) -> dict[str, float] | None:
        """Load reference data from CSV file if provided."""
        if self.reference is None:
            return None
        df = pd.read_csv(self.reference)

        temp_col = df.columns[0]  # First column should be temperature
        cond_col = df.columns[1]  # Second column is conductivity

        reference_data = {
            "temperatures": df[temp_col].values,
            "conductivity": df[cond_col].values,
        }

        reference_unit = sc.Unit(self.reference_units)
        target_unit_sc = sc.Unit("S/m")

        ref_values = sc.array(
            dims=["temperature"],
            values=reference_data["conductivity"],
            unit=reference_unit,
        )
        converted_values = sc.to_unit(ref_values, target_unit_sc)

        converted_data = {
            "temperatures": reference_data["temperatures"],
            "conductivity": converted_values.values,
        }

        return converted_data

    def _plot_arrhenius(self, td, s) -> None:
        """Create Arrhenius plot with credible intervals and legend."""
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.errorbar(
            1000 / td.coords["temperature"].values,
            td.data.values,
            np.sqrt(td.data.variances),
            marker="o",
            ls="",
            color="k",
            zorder=10,
            label="Conductivity",
            capsize=3,
        )

        reference_data = self._load_reference_data()
        if reference_data is not None:
            ax.scatter(
                1000 / reference_data["temperatures"],
                reference_data["conductivity"],
                marker="x",
                color="orangered",
                zorder=9,
                label="Ref. Data",
                s=50,
            )

        # Format plot
        ax.set_yscale("log")
        ax.set_xlabel(r"$1000T^{-1}$ / K$^{-1}$")
        ax.set_ylabel(r"$\sigma$ / S m$^{-1}$")

        ax.legend()
        ax.set_title("Arrhenius Analysis: Ionic Conductivity")

        # Save figure
        fig.savefig(
            self.figures_path / "arrhenius_conductivity.png", dpi=300, bbox_inches="tight"
        )
