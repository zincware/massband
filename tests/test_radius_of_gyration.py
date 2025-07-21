import os

import pytest

import massband


def test_radius_of_gyration_node(tmp_path, bmim_bf4_vectra):
    """Test the RadiusOfGyration node."""
    os.chdir(tmp_path)
    node = massband.RadiusOfGyration(
        file=bmim_bf4_vectra, structures=["CCCCN1[CH-][CH+]N(C)[CH-]1", "F[B-](F)(F)F"]
    )
    node.run()

    bmim_results = node.results["CCCCN1[CH-][CH+]N(C)[CH-]1"]
    bf4_results = node.results["F[B-](F)(F)F"]
    assert bmim_results["mean"] == pytest.approx(2.600, rel=0.1)
    assert bmim_results["std"] == pytest.approx(0.127, rel=0.1)
    assert bf4_results["mean"] == pytest.approx(1.336, rel=0.1)
    assert bf4_results["std"] == pytest.approx(0.012, rel=0.1)
