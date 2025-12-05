from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipp as sc
import zntrack
from kinisi.yeh_hummer import YehHummer as KinisiYehHummerAnalyzer

from massband.diffusion.types import DiffusionData
from massband.utils import sanitize_structure_name


class KinisiYehHummer(zntrack.Node):
    """Perform Yeh-Hummer finite-size correction for diffusion coefficients.

    Analyzes diffusion coefficients from simulations at different box sizes to
    extrapolate the infinite-size diffusion coefficient and estimate viscosity.

    Parameters
    ----------
    data : list[dict[str, DiffusionData]]
        List where each entry corresponds to a box size. Each dict maps structure
        names to their DiffusionData at that box size.
    temperature : float
        Simulation temperature in Kelvin.
    D_infinite_bound : tuple[float, float]
        Lower and upper bounds for infinite-size diffusion coefficient prior in cm^2/s.
    viscosity_bound : tuple[float, float]
        Lower and upper bounds for viscosity prior in mPa*s.

    Attributes
    ----------
    figures_path : Path
        Output directory for Yeh-Hummer plots.
    data_path : Path
        Output directory for MCMC sample data.
    D_infinite : dict[str, DiffusionData]
        Infinite-size diffusion coefficient for each structure.
    viscosity : dict[str, DiffusionData]
        Viscosity for each structure.

    References
    ----------
    .. [1] Yeh, I.-C., & Hummer, G. (2004). J. Phys. Chem. B, 108(15), 4572-4574.
    .. [2] https://kinisi.readthedocs.io/
    """

    data: list[dict[str, DiffusionData]] = zntrack.deps()
    temperature: float = zntrack.params()
    D_infinite_bound: tuple[float, float] = zntrack.params()
    viscosity_bound: tuple[float, float] = zntrack.params()

    figures_path: Path = zntrack.outs_path(zntrack.nwd / "figures")
    data_path: Path = zntrack.outs_path(zntrack.nwd / "data")

    D_infinite: dict[str, DiffusionData] = zntrack.metrics()
    viscosity: dict[str, DiffusionData] = zntrack.metrics()

    def run(self):
        """Execute Yeh-Hummer analysis for all structures."""
        self.figures_path.mkdir(parents=True, exist_ok=True)
        self.data_path.mkdir(parents=True, exist_ok=True)

        self.D_infinite = {}
        self.viscosity = {}

        # Find all unique structures across all box sizes
        all_structures = set()
        for box_data in self.data:
            all_structures.update(box_data.keys())

        # Process each structure
        for structure in all_structures:
            # Extract data for this structure across all box sizes
            D_values = []
            D_errors = []
            box_lengths = []

            for box_data in self.data:
                if structure not in box_data:
                    continue

                diff_data = box_data[structure]
                D_values.append(diff_data["mean"])
                D_errors.append(np.sqrt(diff_data["var"]))

                # Extract box length from box array
                box = np.array(diff_data["box"])
                box_length = np.mean(np.diag(box))
                box_lengths.append(box_length)

            # Skip if not enough data points
            if len(D_values) < 2:
                continue

            # Convert to scipp DataArray
            unit_str = None
            for box_data in self.data:
                if structure in box_data:
                    unit_str = box_data[structure]["unit"]
                    break

            td = sc.DataArray(
                data=sc.array(
                    dims=["system"],
                    values=D_values,
                    variances=np.array(D_errors) ** 2,
                    unit=unit_str,
                ),
                coords={
                    "box_length": sc.Variable(
                        dims=["system"], values=box_lengths, unit="angstrom"
                    )
                },
            )

            # Create kinisi YehHummer object with bounds
            temp = sc.scalar(value=self.temperature, unit="K")
            bounds = (
                (
                    self.D_infinite_bound[0] * sc.Unit("cm^2/s"),
                    self.D_infinite_bound[1] * sc.Unit("cm^2/s"),
                ),
                (
                    self.viscosity_bound[0] * sc.Unit("mPa*s"),
                    self.viscosity_bound[1] * sc.Unit("mPa*s"),
                ),
            )
            yh = KinisiYehHummerAnalyzer(td, temperature=temp, bounds=bounds)

            # Run MCMC
            yh.mcmc()

            # Save MCMC samples
            safe_structure = sanitize_structure_name(structure)
            yh.data_group.save_hdf5(self.data_path / f"{safe_structure}_mcmc_samples.h5")

            # Store summary statistics as DiffusionData
            D_infinite_std = float(sc.std(yh.D_infinite, ddof=1).value)
            self.D_infinite[structure] = {
                "mean": float(sc.mean(yh.D_infinite).value),
                "std": D_infinite_std,
                "var": D_infinite_std**2,
                "occurrences": len(D_values),
                "unit": str(yh.D_infinite.unit),
                "box": None,
            }
            viscosity_std = float(sc.std(yh.shear_viscosity, ddof=1).value)
            self.viscosity[structure] = {
                "mean": float(sc.mean(yh.shear_viscosity).value),
                "std": viscosity_std,
                "var": viscosity_std**2,
                "occurrences": len(D_values),
                "unit": str(yh.shear_viscosity.unit),
                "box": None,
            }

            # Create plot
            self._plot_yeh_hummer(yh, structure)

    def _plot_yeh_hummer(self, yh: KinisiYehHummerAnalyzer, structure: str) -> None:
        """Create Yeh-Hummer plot with credible intervals."""
        safe_structure = sanitize_structure_name(structure)
        fig, ax = plt.subplots()

        credible_intervals = [[16, 84], [2.5, 97.5], [0.15, 99.85]]
        alpha = [0.6, 0.4, 0.2]

        inv_L_data = 1 / yh.box_lengths.values
        max_inv_L = np.max(inv_L_data)
        inv_L_extended = np.linspace(0, max_inv_L * 1.1, 50)

        ax.errorbar(
            inv_L_data,
            yh.diffusion.values,
            np.sqrt(yh.diffusion.variances),
            marker="o",
            ls="",
            color="k",
            zorder=10,
        )

        D_0_samples = yh.data_group["D_0"].values
        eta_samples = yh.data_group["viscosity"].values
        n_samples = len(D_0_samples)
        predictions_extended = np.zeros((len(inv_L_extended), n_samples))

        for i in range(len(yh.data_group["D_0"])):
            slope = yh.viscosity_to_slope(eta_samples[i] * yh.parameter_units[1])
            predictions_extended[:, i] = yh.yeh_hummer_linear(
                inv_L_extended, D_0_samples[i], slope
            )

        for i, ci in enumerate(credible_intervals):
            ax.fill_between(
                inv_L_extended,
                *np.percentile(predictions_extended, ci, axis=1),
                alpha=alpha[i],
                color="#0173B2",
                lw=0,
            )

        ax.set_xlabel(f"1 / L / {yh.box_lengths.unit}")
        ax.set_ylabel(f"D  / {yh.diffusion.unit}")
        ax.set_title(f"Yeh-Hummer Analysis: {structure}")

        fig.savefig(
            self.figures_path / f"{safe_structure}.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)
