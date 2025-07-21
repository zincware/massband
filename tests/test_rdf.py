import os

import numpy as np

from massband.rdf import RadialDistributionFunction


def test_rdf_node_smiles(tmp_path, bmim_bf4_vectra):
    os.chdir(tmp_path)
    node = RadialDistributionFunction(
        file=bmim_bf4_vectra,
        batch_size=5,
        structures=["CCCCN1C=C[N+](=C1)C", "[B-](F)(F)(F)F"],
    )
    node.run()
    assert len(node.results[("CCCCN1C=C[N+](=C1)C|[B-](F)(F)(F)F")]) == 171
    assert len(node.results[("[B-](F)(F)(F)F|[B-](F)(F)(F)F")]) == 171
    assert len(node.results[("CCCCN1C=C[N+](=C1)C|CCCCN1C=C[N+](=C1)C")]) == 171
    assert len(node.results) == 3

    assert (node.figures / "rdf.png").exists()


def test_rdf_node_full(tmp_path, bmim_bf4_vectra):
    os.chdir(tmp_path)
    node = RadialDistributionFunction(
        file=bmim_bf4_vectra,
        batch_size=5,
        structures=None,
    )
    node.run()
    ALL_KEYS = [
        "H|H",
        "H|B",
        "H|C",
        "H|N",
        "H|F",
        "B|B",
        "B|C",
        "B|N",
        "B|F",
        "C|C",
        "C|N",
        "C|F",
        "N|N",
        "N|F",
        "F|F",
    ]
    for key in ALL_KEYS:
        assert len(node.results[key]) == 171
        assert sum(node.results[key]) > 0

    assert len(node.results) == 15


def test_rdf_hh_goes_to_one(tmp_path, bmim_bf4_vectra):
    os.chdir(tmp_path)
    node = RadialDistributionFunction(
        file=bmim_bf4_vectra,
        batch_size=5,
        structures=None,
    )
    node.run()
    hh_rdf = np.array(node.results["H|H"])
    # Check the average of the last 20 bins
    assert np.isclose(np.mean(hh_rdf[-20:]), 1.0, atol=0.1)
