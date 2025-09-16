from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipp as sc
import zntrack
from kinisi.arrhenius import Arrhenius

from massband.diffusion.kinisi_diffusion import KinisiSelfDiffusion


class KinisiDiffusionArrhenius(zntrack.Node):
    """Perform Arrhenius analysis on diffusion coefficients from multiple temperatures.

    Analyzes temperature-dependent diffusion coefficients to extract activation energies
    and pre-exponential factors using Bayesian inference via the kinisi Arrhenius analyzer.

    Parameters
    ----------
    diff : list[KinisiSelfDiffusion]
        List of diffusion analysis nodes at different temperatures.
    temperatures : list[float]
        Corresponding temperatures in Kelvin for each diffusion analysis.
    activation_energy_bound : tuple[float, float]
        Lower and upper bounds for activation energy prior in eV.
    pre_exponential_factor_bound : tuple[float, float]
        Lower and upper bounds for pre-exponential factor prior.
    reference : str | Path | None, default=None
        Path to CSV file containing reference diffusion data. First row should contain
        'temperature' followed by species names, subsequent rows contain temperature
        values and corresponding diffusion coefficients.
    reference_units : str, default="cm^2/s"
        Units of the reference diffusion coefficients for unit conversion.

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
    ...     arrhenius = massband.KinisiArrhenius(
    ...         diff=[diff_300K, diff_350K, diff_400K],
    ...         temperatures=[300, 350, 400],
    ...         activation_energy_bound=[0.1, 2.0],
    ...         pre_exponential_factor_bound=[1e-6, 1e-2],
    ...     )
    >>> project.repro()
    >>> arrhenius.activation_energy["[Li+]"]["mean"]
    0.45
    """

    diff: list[KinisiSelfDiffusion] = zntrack.deps()
    temperatures: list[float] = zntrack.params()
    figures_path: Path = zntrack.outs_path(zntrack.nwd / "figures")
    activation_energy_bound: tuple[float, float] = zntrack.params()
    pre_exponential_factor_bound: tuple[float, float] = zntrack.params()
    reference: str | Path | None = zntrack.deps_path()
    reference_units: str = zntrack.params("cm^2/s")

    activation_energy: dict[str, float] = zntrack.metrics()
    pre_exponential_factor: dict[str, float] = zntrack.metrics()

    def run(self):
        self.figures_path.mkdir(parents=True, exist_ok=True)
        self.activation_energy = {}
        self.pre_exponential_factor = {}

        # Get all unique structures from all diffusion analyses
        all_structures = set()
        for diff_node in self.diff:
            all_structures.update(diff_node.diffusion.keys())

        # Process each structure separately
        for structure in all_structures:
            # Extract diffusion data for this structure across all temperatures
            D = {
                "mean": [x.diffusion[structure]["mean"] for x in self.diff],
                "var": [x.diffusion[structure]["var"] for x in self.diff],
            }

            td = sc.DataArray(
                data=sc.array(
                    dims=["temperature"],
                    values=D["mean"],
                    variances=D["var"],
                    unit=sc.Unit("cm^2/s"),
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

        temp_col = df.columns[0]  # First column should be temperature
        species_cols = df.columns[1:]  # Remaining columns are species

        reference_data = {}
        for species in species_cols:
            reference_data[species] = {
                "temperatures": df[temp_col].values,
                "diffusion": df[species].values,
            }

        reference_unit = sc.Unit(self.reference_units)
        target_unit_sc = sc.Unit("cm^2/s")  # hardcoded for now

        converted_data = {}
        for species, data in reference_data.items():
            ref_values = sc.array(
                dims=["temperature"], values=data["diffusion"], unit=reference_unit
            )
            converted_values = sc.to_unit(ref_values, target_unit_sc)

            converted_data[species] = {
                "temperatures": data["temperatures"],
                "diffusion": converted_values.values,
            }

        return converted_data

    def _plot_arrhenius(self, td, s, structure: str) -> None:
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
            label="Diff. Coef.",
            capsize=3,
        )

        reference_data = self._load_reference_data()
        if reference_data is not None and structure in reference_data:
            ref_data = reference_data[structure]
            ax.scatter(
                1000 / ref_data["temperatures"],
                ref_data["diffusion"],
                marker="x",
                color="orangered",
                zorder=9,
                label="Ref. Data",
                s=50,
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
        ax.set_ylabel(r"$D$ / cm$^2$s$^{-1}$")

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
            self.figures_path / f"arrhenius_{structure}.png", dpi=300, bbox_inches="tight"
        )
