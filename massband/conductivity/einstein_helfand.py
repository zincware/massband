import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zntrack
from kinisi.diffusion import MSCDBootstrap
from kinisi.parser import ASEParser
from rdkit import Chem
from tqdm import tqdm

from massband.dataloader import TimeBatchedLoader


@dataclass
class ConductivityPlotData:
    structure: str
    dt: np.ndarray
    mscd: np.ndarray
    mscd_std: np.ndarray
    distribution: np.ndarray
    sigma_samples: np.ndarray
    sigma_n: float
    start_dt: float


import ase


class KinisiEinsteinHelfandIonicConductivity(zntrack.Node):
    file: str | Path = zntrack.deps_path()
    structures: list[str] = zntrack.params(default_factory=list)
    start_dt: float = zntrack.params(10)  # in ps - start time for conductivity analysis
    start: int = zntrack.params(0)  # in ps - start time for diffusion analysis

    data_path: Path = zntrack.outs_path(zntrack.nwd / "conductivity_data")

    def run(self):
        self.data_path.mkdir(exist_ok=True, parents=True)

        charge_mapping = {}
        for structure in self.structures:
            mol = Chem.MolFromSmiles(structure)
            if mol is not None:
                charge = Chem.GetFormalCharge(mol)
                if charge != 0:
                    charge_mapping[structure] = charge
            else:
                raise ValueError(f"Invalid SMILES string: {structure}")

        print("Charge mapping:", charge_mapping)

        loader = TimeBatchedLoader(
            file=self.file,
            structures=list(charge_mapping),
            wrap=False,
            start=self.start,
        )
        traj = list(tqdm(loader, desc="Loading trajectory", unit="frame"))
        volume = loader.first_frame_atoms.get_volume()

        positions = defaultdict(list)
        for frame in traj:
            for structure in charge_mapping:
                positions[structure].append(frame["position"][structure])

        positions = {
            structure: np.concatenate(positions[structure]) for structure in positions
        }

        length = len(next(iter(positions.values())))

        for key in positions:
            print(f"Positions for {key}: {positions[key].shape}")

        frames = []
        # The charge on the mobile ions, either an array with a value for each ion or a scalar if all values are the same. Optional, default is 1.
        ionic_charge = []
        for key in charge_mapping:
            ionic_charge.extend(positions[key].shape[1] * [charge_mapping[key]])
        print("Ionic charges:", ionic_charge)
        for frame_idx in range(length):
            # create an ase atoms object for each frame
            numbers = []
            frame_positions = []
            for specie_idx, key in enumerate(positions):
                frame_positions.extend(positions[key][frame_idx])
                numbers.extend([specie_idx] * len(positions[key][frame_idx]))

            frame_positions = np.array(frame_positions)

            atoms = ase.Atoms(positions=frame_positions, cell=(100000, 100000, 1000000))
            frames.append(atoms)

        diff = ASEParser(
            atoms=frames,
            specie="X",
            # time_step=effective_time_step,
            # step_skip=self.sampling_rate,
            time_step=0.5 / 1000,
            step_skip=2000,
        )
        ionic_charge = np.array(ionic_charge)

        # diffusion.MSCDBootstrap(cond_anal._delta_t, cond_anal._disp_3d, ionic_charge, cond_anal._n_o,
        #   **uncertainty_params)

        print(f"Diffusion parameters: {ionic_charge.shape =}")
        bootstrap = MSCDBootstrap(diff.delta_t, diff.disp_3d, ionic_charge, diff._n_o)
        bootstrap.conductivity(self.start_dt, 300, volume)

        print(f"{bootstrap.D_J = }")
        print(f"{bootstrap.n = }")
        print(f"{bootstrap.s = }")
        print(f"{bootstrap.sigma = }")

        # Calculate ionic conductivity and confidence interval
        if bootstrap.sigma is not None:
            sigma_samples = (
                bootstrap.sigma.samples
                if hasattr(bootstrap.sigma, "samples")
                else bootstrap.sigma
            )
            sigma_mean = np.mean(sigma_samples)
            sigma_std = np.std(sigma_samples)
            sigma_95_ci = np.percentile(sigma_samples, [2.5, 97.5])

            print("\nIonic Conductivity Results:")
            print(f"Ionic Conductivity: {sigma_mean:.6f} ± {sigma_std:.6f} mS/cm")
            print(
                f"95% Confidence Interval: [{sigma_95_ci[0]:.6f}, {sigma_95_ci[1]:.6f}] mS/cm"
            )

            # Create distribution for plotting (MSCD vs time with uncertainty)
            if hasattr(bootstrap, "gradient") and hasattr(bootstrap, "intercept"):
                gradient_samples = (
                    bootstrap.gradient.samples
                    if hasattr(bootstrap.gradient, "samples")
                    else bootstrap.gradient
                )
                intercept_samples = (
                    bootstrap.intercept.samples
                    if hasattr(bootstrap.intercept, "samples")
                    else bootstrap.intercept
                )
                distribution = (
                    gradient_samples * bootstrap.dt[:, np.newaxis] + intercept_samples
                )
            else:
                # Fallback: create a simple distribution based on MSCD values
                distribution = np.array([bootstrap.n] * len(sigma_samples)).T

            # Store data for plotting
            result = ConductivityPlotData(
                structure="system",
                dt=np.asarray(bootstrap.dt),
                mscd=np.asarray(bootstrap.n),
                mscd_std=np.asarray(bootstrap.s),
                distribution=np.asarray(distribution),
                sigma_samples=np.asarray(sigma_samples),
                sigma_n=float(sigma_mean),
                start_dt=self.start_dt,
            )

            # Save data
            with open(self.data_path / "system.pkl", "wb") as f:
                pickle.dump(result, f)

            # Generate plots
            self.plot()
        else:
            print("\nWarning: No conductivity data available")

    def plot(self):
        """Generate plots for ionic conductivity analysis."""
        credible_intervals = [[16, 84], [2.5, 97.5], [0.15, 99.85]]
        alpha = [0.6, 0.4, 0.2]

        for pkl_path in self.data_path.glob("*.pkl"):
            with open(pkl_path, "rb") as f:
                data: ConductivityPlotData = pickle.load(f)

            # MSCD with std
            fig, ax = plt.subplots()
            ax.errorbar(data.dt, data.mscd, data.mscd_std)
            ax.set_ylabel("MSCD/Å$^2$")
            ax.set_xlabel(r"$\Delta t$/ps")
            ax.set_title(f"{data.structure} Mean Squared Charge Displacement with std")
            fig.savefig(self.data_path / f"{data.structure}_mscd_std.png", dpi=300)
            plt.close(fig)

            # MSCD with credible intervals
            fig, ax = plt.subplots()
            ax.plot(data.dt, data.mscd, "k-")
            for i, ci in enumerate(credible_intervals):
                low, high = np.percentile(data.distribution, ci, axis=1)
                ax.fill_between(data.dt, low, high, alpha=alpha[i], color="#0173B2", lw=0)
            # Add vertical line for start_dt
            ax.axvline(
                data.start_dt, c="k", ls="--", label=f"start_dt = {data.start_dt} ps"
            )
            ax.set_ylabel("MSCD/Å$^2$")
            ax.set_xlabel(r"$\Delta t$/ps")
            ax.set_title(f"{data.structure} MSCD credible intervals")
            ax.legend()
            fig.savefig(
                self.data_path / f"{data.structure}_mscd_credible_intervals.png", dpi=300
            )
            plt.close(fig)

            # Conductivity histogram
            fig, ax = plt.subplots()
            ax.hist(
                data.sigma_samples,
                density=True,
                bins=50,
                color="lightblue",
                edgecolor="k",
            )
            ax.axvline(data.sigma_n, c="red", ls="--", label="Mean (σ_n)")

            # Add confidence intervals
            ci68 = np.percentile(data.sigma_samples, [16, 84])
            ci95 = np.percentile(data.sigma_samples, [2.5, 97.5])
            ax.axvline(ci68[0], c="blue", ls=":", label="68% CI")
            ax.axvline(ci68[1], c="blue", ls=":")

            ax.set_xlabel("σ/mS cm$^{-1}$")
            ax.set_ylabel("p(σ)/mS cm$^{-1}$")
            ax.set_title(f"{data.structure} Ionic Conductivity Histogram")
            ax.legend()

            # Annotate statistics
            sigma_std = np.std(data.sigma_samples)
            textstr = "\n".join(
                (
                    f"Mean: {data.sigma_n:.6f} mS/cm",
                    f"Std: ±{sigma_std:.6f} mS/cm",
                    f"68% CI: [{ci68[0]:.6f}, {ci68[1]:.6f}] mS/cm",
                    f"95% CI: [{ci95[0]:.6f}, {ci95[1]:.6f}] mS/cm",
                )
            )
            ax.text(
                0.95,
                0.95,
                textstr,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
                fontsize=8,
            )

            fig.savefig(
                self.data_path / f"{data.structure}_conductivity_hist.png", dpi=300
            )
            plt.close(fig)
