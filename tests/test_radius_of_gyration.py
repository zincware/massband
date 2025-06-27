import pytest
import ase.build
import znh5md
from massband.radius_of_gyration import RadiusOfGyration


@pytest.fixture
def dummy_h5md_file(tmp_path):
    """Create a dummy H5MD trajectory file for testing."""
    atoms = ase.build.molecule("H2O")
    atoms.cell = [10, 10, 10]  # Set a dummy cell
    atoms.pbc = True

    file_path = tmp_path / "test_traj.h5"
    znh5md.write(file_path, [atoms] * 5)  # Write 5 frames
    return file_path


def test_radius_of_gyration_node(dummy_h5md_file, tmp_path):
    """Test the RadiusOfGyration node."""
    node = RadiusOfGyration(file=dummy_h5md_file, figures=tmp_path / "figures")
    node.run()

    # Check if the output directory and files are created
    assert (tmp_path / "figures").exists()
    assert len(list((tmp_path / "figures").glob("*.png"))) > 0
