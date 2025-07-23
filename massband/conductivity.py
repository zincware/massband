import ase
import h5py
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

    def run(self):
        with self.diffusion.state.fs.open(self.diffusion.file, "rb") as f:
            with h5py.File(f) as file:
                atoms: ase.Atoms = znh5md.IO(file_handle=file)[0]

        volume = atoms.get_volume() * ureg.angstrom**3

        # e^2 / (T * kB * V)
        prefactor = (1 * ureg.elementary_charge) ** 2 / (
            self.temperature * ureg.kelvin * ureg.boltzmann_constant * volume
        )
        # Sum contributions from each ion type
        values = []

        for kind, data in self.diffusion.results.items():
            n_ions = data["occurrences"]
            diff = data["diffusion_coefficient"] * ureg.centimeter**2 / ureg.second
            mol = Chem.MolFromSmiles(kind)
            charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
            if charge == 0:
                print(f"Skipping {kind} with no charge")
                continue
            print(f"Using {n_ions} x {kind} with charge {charge} and diffusion {diff}")

            value = diff * (charge) ** 2 * n_ions
            values.append(value)

        sigma_nernst_einst = (prefactor * sum(values)).to("S/m")
        print(f"Computed Nernst-Einstein ionic conductivity: {sigma_nernst_einst}")

        self.metrics = {
            "Nernst-Einstein ionic conductivity": sigma_nernst_einst.magnitude
        }

        print(sigma_nernst_einst)
