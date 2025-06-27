import pytest
import ase.build
import znh5md
from massband.rdf import RadialDistributionFunction as RDF


@pytest.fixture
def dummy_h5md_file(tmp_path):
    """Create a dummy H5MD trajectory file for testing."""
    atoms = ase.build.molecule("H2O")
    atoms.cell = [10, 10, 10]  # Set a dummy cell
    atoms.pbc = True

    file_path = tmp_path / "test_traj.h5"
    znh5md.write(file_path, [atoms] * 5)  # Write 5 frames
    return file_path


def test_rdf_node(dummy_h5md_file, tmp_path):
    """Test the RDF node."""
    node = RDF(file=dummy_h5md_file, batch_size=5)
    node.run()

    # Check if the output file is created in the node's output directory
    assert (node.figures / "rdf_plot.png").exists()
