import zntrack
from pathlib import Path
from massband.dataloader import TimeBatchedLoader
from tqdm import tqdm
from collections import defaultdict
from rdkit import Chem
import numpy as np

from kinisi.diffusion import MSCDBootstrap
from kinisi.parser import ASEParser
from kinisi.conductivity_analyzer import ConductivityAnalyzer
from dataclasses import dataclass

@dataclass
class DiffusionPlotData:
    structure: str
    dt: np.ndarray
    msd: np.ndarray
    msd_std: np.ndarray
    distribution: np.ndarray
    D_samples: np.ndarray
    D_n: float

import ase

class KinisiEinsteinHelfandIonicConductivity(zntrack.Node):
    file: str | Path = zntrack.deps_path()
    structures: list[str] = zntrack.params(default_factory=list)

    def run(self):

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
            stop=100,
        )
        traj = list(tqdm(loader, desc="Loading trajectory", unit="frame"))
        volume = loader.first_frame_atoms.get_volume()

        positions = defaultdict(list)
        for frame in traj:
            for structure in charge_mapping:
                positions[structure].append(frame["position"][structure])

        positions = {
            structure: np.concatenate(positions[structure])
            for structure in positions
        }

        length= len(next(iter(positions.values())))

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
        bootstrap = MSCDBootstrap(
            diff.delta_t, diff.disp_3d, ionic_charge, diff._n_o
        )
        bootstrap.conductivity(10, 300, volume)

        print(f"{bootstrap.D_J = }")
        print(f"{bootstrap.n = }")
        print(f"{bootstrap.s = }")
        print(f"{bootstrap.sigma = }")
        
        # Calculate ionic conductivity and confidence interval
        sigma_samples = bootstrap.sigma.samples if hasattr(bootstrap.sigma, 'samples') else bootstrap.sigma
        sigma_mean = np.mean(sigma_samples)
        sigma_std = np.std(sigma_samples)
        sigma_95_ci = np.percentile(sigma_samples, [2.5, 97.5])
        
        print("\nIonic Conductivity Results:")
        print(f"Ionic Conductivity: {sigma_mean:.6f} ± {sigma_std:.6f} mS/cm")
        print(f"95% Confidence Interval: [{sigma_95_ci[0]:.6f}, {sigma_95_ci[1]:.6f}] mS/cm")

        # distribution = (
        #     bootstrap.gradient.samples * bootstrap.dt[:, np.newaxis]
        #     + bootstrap.intercept.samples
        # )

        # result = DiffusionPlotData(
        #     structure="system",
        #     dt=np.asarray(bootstrap.dt),
        #     msd=np.asarray(bootstrap.n),
        #     msd_std=np.asarray(bootstrap.s),
        #     distribution=np.asarray(distribution),
        #     D_samples=np.asarray(bootstrap.D_J.samples),
        #     D_n=float(bootstrap.D_J.n),
        # )

        # print("Conductivity results:", result)

        
        
        # ConductivityAnalyzer.from_ase(
        #     trajectory=frames,
        #     parser_params={
        #         "specie": ...,
        #         "time_step": ...,
        #         "step_skip": ...,
        #     },
        #     ionic_charge=np.array(ionic_charge),
        # )
            
        
        
