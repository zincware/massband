import os

import numpy as np

from massband.rdf import RadialDistributionFunction, generate_sorted_pairs


def test_generate_sorted_pairs():
    structure_names = ["H", "O", "H2O", "CO2", "C"]
    expected_pairs = [
        ("H", "H"),
        ("H", "C"),
        ("H", "O"),
        ("C", "C"),
        ("C", "O"),
        ("O", "O"),
        ("H", "CO2"),
        ("H", "H2O"),
        ("C", "CO2"),
        ("C", "H2O"),
        ("O", "CO2"),
        ("O", "H2O"),
        ("CO2", "CO2"),
        ("CO2", "H2O"),
        ("H2O", "H2O"),
    ]

    result = generate_sorted_pairs(structure_names)

    assert result == expected_pairs


def test_rdf_node_smiles(tmp_path, ec_emc, ec_emc_smiles):
    os.chdir(tmp_path)
    node = RadialDistributionFunction(
        file=ec_emc,
        structures=ec_emc_smiles,
        stop=512,
    )
    node.run()

    expected = [f"{a}|{b}" for a, b in generate_sorted_pairs(ec_emc_smiles)]

    assert set(node.results) == set(expected)
    for smile in expected:
        assert len(node.results[smile]) == 285

    assert (node.figures / "rdf.png").exists()


# def test_rdf_node_full(tmp_path, bmim_bf4_vectra):
#     os.chdir(tmp_path)
#     node = RadialDistributionFunction(
#         file=bmim_bf4_vectra,
#         batch_size=5,
#         structures=None,
#     )
#     node.run()
#     ALL_KEYS = [
#         "H|H",
#         "H|B",
#         "H|C",
#         "H|N",
#         "H|F",
#         "B|B",
#         "B|C",
#         "B|N",
#         "B|F",
#         "C|C",
#         "C|N",
#         "C|F",
#         "N|N",
#         "N|F",
#         "F|F",
#     ]
#     for key in ALL_KEYS:
#         assert len(node.results[key]) == 171
#         assert sum(node.results[key]) > 0

#     assert len(node.results) == 15


def test_rdf_hh_goes_to_one(tmp_path, ec_emc):
    os.chdir(tmp_path)
    node = RadialDistributionFunction(
        file=ec_emc,
        stop=512,
        structures=None,
    )
    node.run()
    hh_rdf = np.array(node.results["H|H"])
    # Check the average of the last 20 bins
    assert np.isclose(np.mean(hh_rdf[-20:]), 1.0, atol=0.1)
