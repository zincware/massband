# conftest.py
import pytest
from ase import units
from ase.lattice.cubic import FaceCenteredCubic
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.calculators.emt import EMT
import znh5md

import massband


@pytest.fixture
def emt_md(tmp_path):
    """Generate a fake MD trajectory by rattling atoms on a grid."""

    # TODO: need better data, e.g. containing molecules

    size = 3

    atoms = FaceCenteredCubic(
        directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        symbol="Cu",
        size=(size, size, size),
        pbc=True,
    )

    atoms.calc = EMT()

    MaxwellBoltzmannDistribution(atoms, temperature_K=300)

    dyn = VelocityVerlet(atoms, 5 * units.fs)  # 5 fs time step.

    io = znh5md.IO(
        filename=tmp_path / "md.h5",
        variable_shape=False,
    )

    def save_atoms():
        io.append(atoms)

    dyn.attach(save_atoms, interval=10)
    dyn.run(100)
    return tmp_path / "md.h5"


def test_RadialDistributionFunction(emt_md):
    """Test the radial distribution function calculation."""
    rdf = massband.RadialDistributionFunction(
        file=emt_md,
        batch_size=10,
    )

    # Calculate RDF
    rdf.run()

    # # Check if the result is a numpy array
    # assert isinstance(rdf.result, np.ndarray)

    # # Check if the shape is correct
    # assert rdf.result.shape == (int(5.0 / 0.1) + 1,)

    # # Check if the values are non-negative
    # assert np.all(rdf.result >= 0)
