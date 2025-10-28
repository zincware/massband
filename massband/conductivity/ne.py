import numpy as np
import pint
import zntrack
from rdkit import Chem

from massband.diffusion.types import DiffusionData

ureg = pint.UnitRegistry()


class NernstEinsteinIonicConductivity(zntrack.Node):
    """Compute the ionic conductivity using the Nernst-Einstein equation.

    Parameters
    ----------
    diffusion : dict[str, DiffusionData]
        Dictionary mapping structure names (SMILES strings) to their DiffusionData.
        The box field must be present to calculate volume and number density.
        Provides occurrences (n_i) and box dimensions for all species.
    d_infinite : dict[str, DiffusionData] | None
        Optional dictionary of infinite-size diffusion coefficients (e.g., from
        YehHummer finite-size correction). If provided, these D values override
        those in diffusion, while number densities are still computed from diffusion.
    temperature : float
        Simulation temperature in Kelvin.

    Attributes
    ----------
    conductivity : dict[str, DiffusionData]
        Dictionary mapping "total" to the total ionic conductivity statistics.
        Contains mean, std, var, occurrences (total charged species), unit, and box.

    References
    ----------
    https://www.sciencedirect.com/science/article/pii/S2772422024000120
    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-023-00687-y
    https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.122.136001
    """

    diffusion: dict[str, DiffusionData] = zntrack.deps()
    d_infinite: dict[str, DiffusionData] | None = zntrack.deps(None)
    temperature: float = zntrack.params()
    conductivity: dict[str, DiffusionData] = zntrack.metrics()

    def run(self):
        # Extract volume from box field
        box = None
        for data in self.diffusion.values():
            if data["box"] is not None:
                box = data["box"]
                break

        if box is None:
            raise ValueError(
                "Box information is required in DiffusionData to calculate volume. "
                "Ensure the diffusion analysis includes box data."
            )

        # Calculate volume from box (assuming box is 3x3 cell array)
        box_array = np.array(box)
        volume = np.linalg.det(box_array) * ureg.angstrom**3

        # e^2 / (T * kB) - prefactor without volume (using number density formulation)
        elementary_charge = 1 * ureg.elementary_charge
        boltzmann_constant = 1 * ureg.boltzmann_constant

        prefactor = elementary_charge**2 / (
            self.temperature * ureg.kelvin * boltzmann_constant
        )

        # Collect charged species data for uncertainty propagation
        charged_species = {}

        for kind, data in self.diffusion.items():
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
            # Compute number density from diffusion data
            n_ions = data["occurrences"]
            number_density = n_ions / volume

            # Use d_infinite if provided, otherwise use diffusion
            if self.d_infinite is not None and kind in self.d_infinite:
                D_data = self.d_infinite[kind]
                print(f"Using D_infinite for {kind}")
            else:
                D_data = data
                print(f"Using D from diffusion for {kind}")

            # Use mean and std from selected diffusion coefficient
            D_mean = D_data["mean"] * ureg(D_data["unit"])
            D_std = D_data["std"] * ureg(D_data["unit"])

            # Contribution to conductivity mean using number density formulation
            # σ_i = ρ_i * q_i^2 * D_i
            contribution = number_density * (charge**2) * D_mean
            contributions.append(contribution)

            # Contribution to conductivity variance (error propagation)
            # Var(c * D) = c^2 * Var(D), where c = ρ_i * charge^2
            c = number_density * (charge**2)
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

        # Calculate total number of charged species
        total_charged_species = sum(
            data["occurrences"] for _, data in charged_species.values()
        )

        print(
            f"Nernst-Einstein ionic conductivity: {sigma.magnitude:.3e} ± {sigma_std.magnitude:.3e} S/m"
        )

        # Store results under "total" key as DiffusionData structure
        self.conductivity = {
            "total": {
                "mean": sigma.magnitude,
                "std": sigma_std.magnitude,
                "var": sigma_variance.magnitude,
                "occurrences": total_charged_species,
                "unit": str(sigma.units),
                "box": box,
            }
        }
