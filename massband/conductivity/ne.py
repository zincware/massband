import pint
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
    conductivity: dict = zntrack.metrics()

    def run(self):
        atoms = self.diffusion.frames[0]
        volume = atoms.get_volume() * ureg.angstrom**3

        # e^2 / (T * kB * V)
        elementary_charge = 1 * ureg.elementary_charge
        boltzmann_constant = 1 * ureg.boltzmann_constant

        prefactor = elementary_charge**2 / (
            self.temperature * ureg.kelvin * boltzmann_constant * volume
        )

        # Collect charged species data for uncertainty propagation
        charged_species = {}

        for kind, data in self.diffusion.diffusion.items():
            mol = Chem.MolFromSmiles(kind)
            if mol is None:
                raise ValueError(f"Invalid SMILES string: {kind}")

            charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
            if charge == 0:
                print(f"Skipping {kind} with no charge")
                continue
            charged_species[kind] = (charge, data)
            print(f"Using {data['occurrences']} x {kind} with charge {charge}")

        if not charged_species:
            raise ValueError("No charged species found for conductivity calculation")

        # Calculate the mean and standard deviation using error propagation
        contributions = []
        variances = []

        for kind, (charge, data) in charged_species.items():
            # Use mean and std from diffusion coefficient
            D_mean = data["mean"] * ureg(data["unit"])
            D_std = data["std"] * ureg(data["unit"])
            n_ions = data["occurrences"]

            # Contribution to conductivity mean
            contribution = D_mean * (charge**2) * n_ions
            contributions.append(contribution)

            # Contribution to conductivity variance (error propagation)
            # Var(c * D) = c^2 * Var(D), where c = charge^2 * n_ions
            c = (charge**2) * n_ions
            contribution_variance = (c**2) * (D_std**2)
            variances.append(contribution_variance)

        # Convert to conductivity units
        total_contribution = sum(contributions)
        total_variance = sum(variances)

        sigma = (prefactor * total_contribution).to(ureg.siemens / ureg.meter)
        sigma_variance = (prefactor**2 * total_variance).to(
            (ureg.siemens / ureg.meter) ** 2
        )
        sigma_std = sigma_variance**0.5

        print(
            f"Nernst-Einstein ionic conductivity: {sigma.magnitude:.3e} ± {sigma_std.magnitude:.3e} S/m"
        )

        # Store results
        self.conductivity = {
            "mean": sigma.magnitude,
            "std": sigma_std.magnitude,
            "var": sigma_variance.magnitude,
            "unit": str(sigma.units),
        }
