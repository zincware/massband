import ase
import h5py
import numpy as np
import scipp as sc
import znh5md
import zntrack
from rdkit import Chem

from massband.diffusion.kinisi_diffusion import KinisiSelfDiffusion


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
    conductivity: dict = zntrack.metrics()

    def run(self):
        atoms = self.diffusion.frames[0]
        volume = atoms.get_volume() * sc.Unit("angstrom**3")

        # e^2 / (T * kB * V)
        elementary_charge = 1.602176634e-19 * sc.Unit("coulomb")  # elementary charge
        boltzmann_constant = 1.380649e-23 * sc.Unit("J/K")  # Boltzmann constant

        prefactor = elementary_charge**2 / (
            self.temperature * sc.Unit("K") * boltzmann_constant * volume
        )

        # Collect charged species data for uncertainty propagation
        species_data = {}
        charged_species = []

        for kind, data in self.diffusion.diffusion.items():
            mol = Chem.MolFromSmiles(kind)
            charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
            if charge == 0:
                print(f"Skipping {kind} with no charge")
                continue

            species_data[kind] = {
                "charge": charge,
                "n_ions": data["occurrences"],
                "D_mean": data["mean"],
                "D_std": data["std"],
            }
            charged_species.append(kind)
            print(f"Using {data['occurrences']} x {kind} with charge {charge}")

        if not charged_species:
            raise ValueError("No charged species found for conductivity calculation")

        # Normal uncertainty propagation
        # Calculate the mean and standard deviation using error propagation
        sum_contribution = sc.scalar(0.0, unit="angstrom**2/ps")
        sum_variance = sc.scalar(0.0, unit="angstrom**4/ps**2")

        for kind in charged_species:
            data = species_data[kind]
            # Use mean and std from diffusion coefficient
            D_mean = data["D_mean"] * sc.Unit("angstrom**2/ps")
            D_std = data["D_std"] * sc.Unit("angstrom**2/ps")
            charge = data["charge"]
            n_ions = data["n_ions"]

            # Contribution to conductivity mean
            contribution = D_mean * (charge**2) * n_ions
            sum_contribution += contribution

            # Contribution to conductivity variance (error propagation)
            # Var(c * D) = c^2 * Var(D), where c = charge^2 * n_ions
            c = (charge**2) * n_ions
            contribution_variance = (c**2) * (D_std**2)
            sum_variance += contribution_variance

        # Convert to conductivity units
        sigma_with_units = prefactor * sum_contribution
        sigma_variance_with_units = prefactor**2 * sum_variance

        # Convert to mS/cm
        sigma_mean = sc.to_unit(sigma_with_units, "S/m").value
        sigma_variance = sc.to_unit(sigma_variance_with_units, "(S/m)**2").value
        sigma_std = np.sqrt(sigma_variance)

        print(
            f"Nernst-Einstein ionic conductivity: {sigma_mean:.3e} ± {sigma_std:.3e} S/m"
        )

        # Store results
        self.conductivity = {
            "mean": sigma_mean,
            "std": sigma_std,
            "var": sigma_variance,
        }
