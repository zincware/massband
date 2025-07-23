import ase
import h5py
import numpy as np
import pickle
import pint
import znh5md
import zntrack
from rdkit import Chem

from massband.diffusion.kinisi_diffusion import KinisiSelfDiffusion

ureg = pint.UnitRegistry()


class NernstEinsteinIonicConductivity(zntrack.Node):
    """Compute the ionic conductivity using the Nernst-Einstein equation.


    References
    ----------
    https://www.sciencedirect.com/science/article/pii/S2772422024000120
    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-023-00687-y
    https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.122.136001
    """

    diffusion: KinisiSelfDiffusion = zntrack.deps()
    temperature: float = zntrack.params()
    metrics: dict = zntrack.metrics()

    def run(self):
        try:
            with self.diffusion.state.fs.open(self.diffusion.file, "rb") as f:
                with h5py.File(f) as file:
                    atoms: ase.Atoms = znh5md.IO(file_handle=file)[0]
        except FileNotFoundError:
            with h5py.File(self.diffusion.file, "r") as file:
                atoms: ase.Atoms = znh5md.IO(file_handle=file)[0]

        volume = atoms.get_volume() * ureg.angstrom**3

        # e^2 / (T * kB * V)
        prefactor = (1 * ureg.elementary_charge) ** 2 / (
            self.temperature * ureg.kelvin * ureg.boltzmann_constant * volume
        )

        # Load D_samples for each species for uncertainty propagation
        species_data = {}
        charged_species = []
        
        for kind, data in self.diffusion.results.items():
            mol = Chem.MolFromSmiles(kind)
            charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
            if charge == 0:
                print(f"Skipping {kind} with no charge")
                continue
                
            # Load D_samples from pickle file
            data_file = self.diffusion.data_path / f"{kind}.pkl"
            if data_file.exists():
                with open(data_file, "rb") as f:
                    diffusion_plot_data = pickle.load(f)
                D_samples = diffusion_plot_data.D_samples
            else:
                # Fallback: create samples from mean and std if pickle not available
                D_mean = data["diffusion_coefficient"]
                D_std = data["std"]
                D_samples = np.random.normal(D_mean, D_std, 1000)
                print(f"Warning: Using normal approximation for {kind} D_samples")
            
            species_data[kind] = {
                "D_samples": D_samples,
                "charge": charge,
                "n_ions": data["occurrences"],
                "D_mean": data["diffusion_coefficient"]
            }
            charged_species.append(kind)
            print(f"Using {data['occurrences']} x {kind} with charge {charge}")

        if not charged_species:
            raise ValueError("No charged species found for conductivity calculation")

        # Monte Carlo uncertainty propagation
        n_samples = min([len(species_data[kind]["D_samples"]) for kind in charged_species])
        conductivity_samples = []

        for i in range(n_samples):
            sample_sum = 0 * ureg.centimeter**2 / ureg.second
            
            for kind in charged_species:
                data = species_data[kind]
                # Sample from diffusion coefficient distribution
                D_sample = data["D_samples"][i] * ureg.centimeter**2 / ureg.second
                charge = data["charge"]
                n_ions = data["n_ions"]
                
                # Contribution to conductivity
                contribution = D_sample * (charge ** 2) * n_ions
                sample_sum += contribution
            
            sigma_sample = (prefactor * sample_sum).to("S/m")
            conductivity_samples.append(sigma_sample.magnitude)

        conductivity_samples = np.array(conductivity_samples)

        # Compute statistics
        sigma_mean = np.mean(conductivity_samples)
        sigma_std = np.std(conductivity_samples)
        sigma_ci68 = np.percentile(conductivity_samples, [16, 84])
        sigma_ci95 = np.percentile(conductivity_samples, [2.5, 97.5])
        
        # Asymmetric uncertainties
        uncertainty_low = sigma_mean - sigma_ci68[0]
        uncertainty_high = sigma_ci68[1] - sigma_mean

        print(f"Nernst-Einstein ionic conductivity: {sigma_mean:.3e} ± {sigma_std:.3e} S/m")
        print(f"68% CI: [{sigma_ci68[0]:.3e}, {sigma_ci68[1]:.3e}] S/m")
        print(f"95% CI: [{sigma_ci95[0]:.3e}, {sigma_ci95[1]:.3e}] S/m")

        # Store comprehensive results
        self.metrics = {
            "Nernst-Einstein ionic conductivity": sigma_mean,
            "conductivity_std": sigma_std,
            "conductivity_ci68": sigma_ci68.tolist(),
            "conductivity_ci95": sigma_ci95.tolist(),
            "conductivity_asymmetric_uncertainty": [uncertainty_low, uncertainty_high],
            "conductivity_samples": conductivity_samples.tolist(),
            "n_monte_carlo_samples": n_samples
        }
