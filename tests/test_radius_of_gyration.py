import os

import pytest

import massband

ec_emc_results = {
    "F[P-](F)(F)(F)(F)F": {"mean": 1.4542316198349, "std": 0.01072658970952034},
    "C1COC(=O)O1": {"mean": 1.4764798879623413, "std": 0.01119674276560545},
    "CCOC(=O)OC": {"mean": 2.0026965141296387, "std": 0.046107202768325806},
    "C1=COC(=O)O1": {"mean": 1.430567741394043, "std": 0.010287800803780556},
    "COC(=O)OC": {"mean": 1.7105259895324707, "std": 0.01731782592833042},
    "global": {"mean": 1.6781843900680542, "std": 0.21587470173835754},
}


def test_radius_of_gyration_node(tmp_path, ec_emc, ec_emc_smiles):
    """Test the RadiusOfGyration node."""
    os.chdir(tmp_path)
    node = massband.RadiusOfGyration(
        file=ec_emc,
        structures=ec_emc_smiles,
        stop=512,
    )
    node.run()

    for key, value in ec_emc_results.items():
        results = node.results[key]
        assert results["mean"] == pytest.approx(value["mean"], rel=0.1)
        assert results["std"] == pytest.approx(value["std"], rel=0.1)
