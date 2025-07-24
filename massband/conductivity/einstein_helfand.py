import pickle
from collections import defaultdict
from pathlib import Path

import ase
import numpy as np
import zntrack
from kinisi.diffusion import MSCDBootstrap
from kinisi.parser import ASEParser
from rdkit import Chem
from tqdm import tqdm

from massband.dataloader import TimeBatchedLoader
from massband.kinisi import KinisiPlotData
from massband.plotting.kinisi import PlottingConfig, plot_kinisi_results


class KinisiEinsteinHelfandIonicConductivity(zntrack.Node):
    file: str | Path = zntrack.deps_path()
    structures: list[str] = zntrack.params(default_factory=list)
    start_dt: float = zntrack.params(10)  # in ps - start time for conductivity analysis
    start: int = zntrack.params(0)  # in ps - start time for diffusion analysis
    time_step: float = zntrack.params(0.5)  # in fs - time step of the simulation
    sampling_rate: int = zntrack.params(1000)  # in fs - sampling

    results: dict = zntrack.metrics()
    data_path: Path = zntrack.outs_path(zntrack.nwd / "conductivity_data")

    def _build_charge_mapping(self) -> dict[str, int]:
        """Build charge mapping from SMILES structures.
        
        Returns
        -------
        dict[str, int]
            Dictionary mapping structure SMILES to their formal charges.
            
        Raises
        ------
        ValueError
            If any SMILES string is invalid.
        """
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
        return charge_mapping

    def _construct_frames(self, positions: dict, charge_mapping: dict[str, int]) -> tuple[list, np.ndarray]:
        """Construct ASE Atoms frames from positions data.
        
        Parameters
        ----------
        positions : dict
            Dictionary mapping structure names to position arrays.
        charge_mapping : dict[str, int]
            Dictionary mapping structure names to formal charges.
            
        Returns
        -------
        tuple[list, np.ndarray]
            Tuple of (frames list, ionic_charge array).
        """
        length = len(next(iter(positions.values())))

        for key in positions:
            print(f"Positions for {key}: {positions[key].shape}")

        frames = []
        ionic_charge = []
        for key in charge_mapping:
            ionic_charge.extend(positions[key].shape[1] * [charge_mapping[key]])
        print("Ionic charges:", ionic_charge)
        
        for frame_idx in range(length):
            numbers = []
            frame_positions = []
            for specie_idx, key in enumerate(positions):
                frame_positions.extend(positions[key][frame_idx])
                numbers.extend([specie_idx] * len(positions[key][frame_idx]))

            frame_positions = np.array(frame_positions)
            atoms = ase.Atoms(positions=frame_positions, cell=(100000, 100000, 1000000))
            frames.append(atoms)
            
        return frames, np.array(ionic_charge)

    def _process_results(self, bootstrap) -> None:
        """Process conductivity results and save data.
        
        Parameters
        ----------
        bootstrap : MSCDBootstrap
            MSCDBootstrap object with conductivity results.
        """
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
            print(f"Ionic Conductivity: {sigma_mean:.6f} \u00b1 {sigma_std:.6f} mS/cm")
            print(
                f"95% Confidence Interval: [{sigma_95_ci[0]:.6f}, {sigma_95_ci[1]:.6f}] mS/cm"
            )

            ci68 = np.percentile(sigma_samples, [16, 84])
            uncertainty_low = sigma_mean - ci68[0]
            uncertainty_high = ci68[1] - sigma_mean

            self.results["system"] = {
                "ionic_conductivity": float(sigma_mean),
                "std": float(sigma_std),
                "credible_interval_68": ci68.tolist(),
                "credible_interval_95": sigma_95_ci.tolist(),
                "asymmetric_uncertainty": [uncertainty_low, uncertainty_high],
                "samples": sigma_samples.tolist()
                if isinstance(sigma_samples, np.ndarray)
                else sigma_samples,
            }

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
                distribution = np.array([bootstrap.n] * len(sigma_samples)).T

            plot_data = KinisiPlotData(
                structure="system",
                dt=np.asarray(bootstrap.dt),
                displacement=np.asarray(bootstrap.n),
                displacement_std=np.asarray(bootstrap.s),
                distribution=np.asarray(distribution),
                samples=np.asarray(sigma_samples),
                mean_value=float(sigma_mean),
                start_dt=self.start_dt,
            )

            with open(self.data_path / "system.pkl", "wb") as f:
                pickle.dump(plot_data, f)

            self.plot()
        else:
            print("\nWarning: No conductivity data available")

    def run(self):
        self.data_path.mkdir(exist_ok=True, parents=True)
        self.results = {}

        charge_mapping = self._build_charge_mapping()

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

        frames, ionic_charge = self._construct_frames(positions, charge_mapping)

        diff = ASEParser(
            atoms=frames,
            specie="X",
            time_step=self.time_step / 1000,
            step_skip=self.sampling_rate,
        )

        print(f"Diffusion parameters: {ionic_charge.shape =}")
        bootstrap = MSCDBootstrap(diff.delta_t, diff.disp_3d, ionic_charge, diff._n_o)
        bootstrap.conductivity(self.start_dt, 300, volume)

        self._process_results(bootstrap)

    def plot(self):
        """Generate plots for ionic conductivity analysis.
        
        Notes
        -----
        Creates standard kinisi plots including displacement with std,
        credible intervals, and histogram plots for conductivity data.
        """
        config = PlottingConfig(
            displacement_label="MSCD",
            displacement_unit="\u00c5$^2$",
            value_label="\u03c3",
            value_unit="mS cm$^{-1}$",
            msd_title="system Mean Squared Charge Displacement with std",
            msd_filename="system_mscd_std.png",
            ci_title="system MSCD credible intervals",
            ci_filename="system_mscd_credible_intervals.png",
            hist_title="system Ionic Conductivity Histogram",
            hist_filename="system_conductivity_hist.png",
            hist_label="Mean (\u03c3_n)",
        )
        for pkl_path in self.data_path.glob("*.pkl"):
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            plot_kinisi_results(data, self.data_path, config)
