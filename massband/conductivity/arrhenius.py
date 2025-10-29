from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipp as sc
import zntrack
from kinisi.arrhenius import Arrhenius

from massband.diffusion.types import DiffusionData
from massband.utils import sanitize_structure_name


class KinisiConductivityArrhenius(zntrack.Node):
    """Perform Arrhenius analysis on ionic conductivity coefficients from multiple temperatures.

    Analyzes temperature-dependent ionic conductivity coefficients to extract activation energies
    and pre-exponential factors using Bayesian inference via the kinisi Arrhenius analyzer.

    Parameters
    ----------
    data : list[dict[str, DiffusionData]]
        List of conductivity data dictionaries at different temperatures.
        Each dict maps structure names to their conductivity data (e.g., "total").
    temperatures : list[float]
        Corresponding temperatures in Kelvin for each conductivity analysis.
    activation_energy_bound : tuple[float, float]
        Lower and upper bounds for activation energy prior in eV.
    pre_exponential_factor_bound : tuple[float, float]
        Lower and upper bounds for pre-exponential factor prior.
    reference : str | Path | None, default=None
        Path to CSV file containing reference conductivity data. First row should contain
        'temperature' followed by structure names, subsequent rows contain temperature
        values and corresponding conductivity coefficients.
    reference_units : str, default="S/m"
        Units of the reference conductivity coefficients for unit conversion.

    Attributes
    ----------
    figures_path : Path
        Output directory for Arrhenius plots.
    activation_energy : dict[str, dict[str, float]]
        Activation energies with mean and std for each analyzed structure.
    pre_exponential_factor : dict[str, dict[str, float]]
        Pre-exponential factors with mean and std for each analyzed structure.

    Examples
    --------
    >>> with project:
    ...     cond_300K = massband.NernstEinsteinIonicConductivity(...)
    ...     cond_350K = massband.NernstEinsteinIonicConductivity(...)
    ...     cond_400K = massband.NernstEinsteinIonicConductivity(...)
    ...     arrhenius = massband.KinisiConductivityArrhenius(
    ...         data=[
    ...             cond_300K.conductivity,
    ...             cond_350K.conductivity,
    ...             cond_400K.conductivity,
    ...         ],
    ...         temperatures=[300, 350, 400],
    ...         activation_energy_bound=[0.1, 2.0],
    ...         pre_exponential_factor_bound=[1e-6, 1e-2],
    ...     )
    >>> project.repro()
    >>> arrhenius.activation_energy["total"]["mean"]
    0.45
    """

    data: list[dict[str, DiffusionData]] = zntrack.deps()
    temperatures: list[float] = zntrack.params()
    figures_path: Path = zntrack.outs_path(zntrack.nwd / "figures")
    activation_energy_bound: tuple[float, float] = zntrack.params()
    pre_exponential_factor_bound: tuple[float, float] = zntrack.params()
    reference: str | Path | None = zntrack.deps_path()
    reference_units: str = zntrack.params("S/m")

    activation_energy: dict[str, float] = zntrack.metrics()
    pre_exponential_factor: dict[str, float] = zntrack.metrics()

    def run(self):
        self.figures_path.mkdir(parents=True, exist_ok=True)
        self.activation_energy = {}
        self.pre_exponential_factor = {}

        # Get all unique structures from all conductivity data
        all_structures = set()
        for data_dict in self.data:
            all_structures.update(data_dict.keys())

        # Process each structure separately
        for structure in all_structures:
            # Extract conductivity data for this structure across all temperatures
            sigma = {
                "mean": [data_dict[structure]["mean"] for data_dict in self.data],
                "var": [data_dict[structure]["var"] for data_dict in self.data],
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
                        dims=["temperature"], values=self.temperatures, unit="K"
                    )
                },
            )

            # Perform Arrhenius analysis
            s = Arrhenius(
                td,
                bounds=[
                    [
                        self.activation_energy_bound[0] * sc.Unit("eV"),
                        self.activation_energy_bound[1] * sc.Unit("eV"),
                    ],
                    [
                        self.pre_exponential_factor_bound[0] * td.data.unit,
                        self.pre_exponential_factor_bound[1] * td.data.unit,
                    ],
                ],
            )
            s.mcmc()

            # Store results
            self.activation_energy[structure] = {
                "mean": sc.mean(s.activation_energy).value,
                "std": sc.std(s.activation_energy, ddof=1).value,
            }
            self.pre_exponential_factor[structure] = {
                "mean": sc.mean(s.preexponential_factor).value,
                "std": sc.std(s.preexponential_factor, ddof=1).value,
            }

            # Create Arrhenius plot for this structure
            self._plot_arrhenius(td, s, structure)

    def _load_reference_data(self) -> dict[str, dict[str, float]] | None:
        """Load reference data from CSV file if provided."""
        if self.reference is None:
            return None
        df = pd.read_csv(self.reference)

        # Strip whitespace from column names and filter out unnamed columns
        df.columns = df.columns.str.strip()
        df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]

        temp_col = df.columns[0]  # First column should be temperature
        structure_cols = df.columns[1:]  # Remaining columns are structures

        reference_data = {}
        for structure in structure_cols:
            reference_data[structure] = {
                "temperatures": df[temp_col].values,
                "conductivity": df[structure].values,
            }

        reference_unit = sc.Unit(self.reference_units)
        target_unit_sc = sc.Unit("S/m")

        converted_data = {}
        for structure, data in reference_data.items():
            ref_values = sc.array(
                dims=["temperature"], values=data["conductivity"], unit=reference_unit
            )
            converted_values = sc.to_unit(ref_values, target_unit_sc)

            converted_data[structure] = {
                "temperatures": data["temperatures"],
                "conductivity": converted_values.values,
            }

        return converted_data

    def _plot_arrhenius(self, td, s, structure: str) -> None:
        safe_structure = sanitize_structure_name(structure)
        """Create Arrhenius plot with credible intervals and legend."""
        credible_intervals = [[16, 84], [2.5, 97.5], [0.15, 99.85]]
        alpha = [0.6, 0.4, 0.2]
        sigmas = [1, 2, 3]

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
        if reference_data is not None and structure in reference_data:
            ref_data = reference_data[structure]
            ax.scatter(
                1000 / ref_data["temperatures"],
                ref_data["conductivity"],
                marker="x",
                color="orangered",
                zorder=9,
                label="Ref. Data",
                s=50,
            )
        elif reference_data is not None:
            print(
                f"Warning: Reference data for structure '{structure}' not found in {reference_data.keys()}"
            )

        # Plot credible intervals
        for i, (ci, sigma) in enumerate(zip(credible_intervals, sigmas)):
            ax.fill_between(
                1000 / td.coords["temperature"].values,
                *np.percentile(s.distribution, ci, axis=1),
                alpha=alpha[i],
                color="#0173B2",
                lw=0,
                label=f"±{sigma}σ interval",
            )

        # Format plot
        ax.set_yscale("log")
        ax.set_xlabel(r"$1000T^{-1}$ / K$^{-1}$")
        ax.set_ylabel(r"$\sigma$ / S m$^{-1}$")

        # Add activation energy to lower left
        ea_mean = self.activation_energy[structure]["mean"]
        ea_std = self.activation_energy[structure]["std"]
        ax.text(
            0.05,
            0.05,
            f"$E_a$ = {ea_mean:.3f} ± {ea_std:.3f} eV",
            transform=ax.transAxes,
            verticalalignment="bottom",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

        ax.legend()
        ax.set_title(f"Arrhenius Analysis: {structure}")

        # Save figure
        fig.savefig(
            self.figures_path / f"arrhenius_{safe_structure}.png",
            dpi=300,
            bbox_inches="tight",
        )
