from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipp as sc
import zntrack
from kinisi.arrhenius import Arrhenius

from massband.abc import ComparisonResults
from massband.comparison_utils import create_bar_comparison
from massband.diffusion.types import DiffusionData
from massband.utils import sanitize_structure_name


class KinisiDiffusionArrhenius(zntrack.Node):
    """Perform Arrhenius analysis on diffusion coefficients from multiple temperatures.

    Analyzes temperature-dependent diffusion coefficients to extract activation energies
    and pre-exponential factors using Bayesian inference via the kinisi Arrhenius analyzer.

    Parameters
    ----------
    data : list[dict[str, DiffusionData]]
        List of diffusion data dictionaries at different temperatures.
        Each dict maps structure names to their DiffusionData.
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
    ...     diff_300K = massband.KinisiSelfDiffusion(...)
    ...     diff_350K = massband.KinisiSelfDiffusion(...)
    ...     diff_400K = massband.KinisiSelfDiffusion(...)
    ...     arrhenius = massband.KinisiDiffusionArrhenius(
    ...         data=[diff_300K.diffusion, diff_350K.diffusion, diff_400K.diffusion],
    ...         temperatures=[300, 350, 400],
    ...         activation_energy_bound=[0.1, 2.0],
    ...         pre_exponential_factor_bound=[1e-6, 1e-2],
    ...     )
    >>> project.repro()
    >>> arrhenius.activation_energy["[Li+]"]["mean"]
    0.45
    """

    data: list[dict[str, DiffusionData]] = zntrack.deps()
    temperatures: list[float] = zntrack.params()
    figures_path: Path = zntrack.outs_path(zntrack.nwd / "figures")
    activation_energy_bound: tuple[float, float] = zntrack.params()
    pre_exponential_factor_bound: tuple[float, float] = zntrack.params()
    reference: str | Path | None = zntrack.deps_path()
    reference_units: str = zntrack.params("cm^2/s")

    activation_energy: dict[str, float] = zntrack.metrics()
    pre_exponential_factor: dict[str, float] = zntrack.metrics()

    @staticmethod
    def compare(  # noqa: C901
        *nodes: "KinisiDiffusionArrhenius",
        labels: list[str] | None = None,
        structures: list[str] | None = None,
        use_plotly: bool = False,
    ) -> ComparisonResults:
        """Compare Arrhenius parameters from multiple calculations.

        Parameters
        ----------
        *nodes : KinisiDiffusionArrhenius
            Multiple Arrhenius analysis node instances to compare
        labels : list[str] | None
            Labels for each node (e.g., ["ML potential", "DFT", "Experiment"]).
            If None, uses node indices as labels.
        structures : list[str] | None
            Specific structures to compare (e.g., ["[Li+]", "EC"]).
            If None, compares all common structures across all nodes.
        use_plotly : bool
            If True, creates plotly figures; otherwise matplotlib.

        Returns
        -------
        ComparisonResults
            Dictionary containing comparative Arrhenius plots with keys:
            - "activation_energies": bar chart comparing Ea values
            - "pre_exponential_factors": bar chart comparing A values
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

        # Find common structures across all nodes
        common_structures = set(nodes[0].activation_energy.keys())
        for node in nodes[1:]:
            common_structures &= set(node.activation_energy.keys())

        if not common_structures:
            raise ValueError("No common structures found across all nodes")

        # Filter to specific structures if requested
        if structures is not None:
            common_structures = set(structures) & common_structures
            if not common_structures:
                raise ValueError(
                    f"None of the requested structures {structures} are common to all nodes"
                )

        common_structures = sorted(common_structures)
        figures: dict[str, plt.Figure] = {}

        # Extract Arrhenius parameters
        ea_values: dict[str, list[float]] = {}
        ea_errors: dict[str, list[float]] = {}
        pre_exp_values: dict[str, list[float]] = {}
        pre_exp_errors: dict[str, list[float]] = {}

        for structure in common_structures:
            ea_values[structure] = []
            ea_errors[structure] = []
            pre_exp_values[structure] = []
            pre_exp_errors[structure] = []

            for node in nodes:
                ea_data = node.activation_energy[structure]
                pef_data = node.pre_exponential_factor[structure]

                ea_values[structure].append(ea_data["mean"])
                ea_errors[structure].append(ea_data["std"])
                pre_exp_values[structure].append(pef_data["mean"])
                pre_exp_errors[structure].append(pef_data["std"])

        # Create activation energy comparison
        ea_fig = create_bar_comparison(
            ea_values,
            ea_errors,
            labels,
            title="Activation Energy Comparison",
            ylabel="Ea / eV",
            use_plotly=use_plotly,
        )
        figures["activation_energies"] = ea_fig

        # Create pre-exponential factor comparison (use log scale for values)
        # We'll create this manually to handle log scale better
        if use_plotly:
            import plotly.graph_objects as go

            fig = go.Figure()
            categories = list(pre_exp_values.keys())

            for i, label in enumerate(labels):
                vals = [pre_exp_values[cat][i] for cat in categories]
                errs = [pre_exp_errors[cat][i] for cat in categories]
                fig.add_trace(
                    go.Bar(
                        name=label,
                        x=categories,
                        y=vals,
                        error_y={"type": "data", "array": errs},
                    )
                )

            fig.update_layout(
                title="Pre-Exponential Factor Comparison",
                yaxis_title="A / cm²/s",
                yaxis_type="log",
                barmode="group",
            )
            figures["pre_exponential_factors"] = fig
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            categories = list(pre_exp_values.keys())
            x = np.arange(len(categories))
            width = 0.8 / len(labels)

            for i, label in enumerate(labels):
                vals = [pre_exp_values[cat][i] for cat in categories]
                errs = [pre_exp_errors[cat][i] for cat in categories]
                offset = (i - len(labels) / 2) * width + width / 2
                ax.bar(x + offset, vals, width, label=label, yerr=errs, capsize=5)

            ax.set_yscale("log")
            ax.set_ylabel("A / cm²/s")
            ax.set_title("Pre-Exponential Factor Comparison")
            ax.set_xticks(x)
            ax.set_xticklabels(categories, rotation=45, ha="right")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            figures["pre_exponential_factors"] = fig

        return {"figures": figures}

    def run(self):
        self.figures_path.mkdir(parents=True, exist_ok=True)
        self.activation_energy = {}
        self.pre_exponential_factor = {}

        # Get all unique structures from all diffusion data
        all_structures = set()
        for data_dict in self.data:
            all_structures.update(data_dict.keys())

        # Process each structure separately
        for structure in all_structures:
            # Extract diffusion data for this structure across all temperatures
            D = {
                "mean": [data_dict[structure]["mean"] for data_dict in self.data],
                "var": [data_dict[structure]["var"] for data_dict in self.data],
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
            self.figures_path / f"arrhenius_{safe_structure}.png",
            dpi=300,
            bbox_inches="tight",
        )
