import os

import pytest

import massband

# Reference coordination numbers calculated from EC-EMC test system
# These values were computed using the coordination number analysis with:
# - density_threshold=0.5, max_integration_distance=8.0, bin_width=0.1
# - First 1000 frames of the EC-EMC trajectory
# Values represent the coordination number for each atom pair's first shell
cn = {
    "C|C": 1.7157283243495405,
    "C|F": 1.203568519469787,
    "C|H": 2.595999898919866,
    "C|Li": 0.1169532774884077,
    "C|O": 1.509164533091309,
    "C|P": 0.9018571012333788,
    "F|C": 6.531866652539157,
    "F|F": 4.1382273824495766,
    "F|H": 4.940437836841694,
    "F|Li": 0.21729591176338237,
    "F|O": 1.6775724163344177,
    "F|P": 0.9088631764763263,  # should be 1
    "H|C": 1.479776747633753,
    "H|F": 0.5189081316595214,
    "H|H": 1.7362753527115318,
    "H|Li": 0.1802230210863178,
    "H|O": 0.9087298880516652,
    "H|P": 0.3210217652771648,
    "Li|C": 3.808291098216276,
    "Li|F": 1.3037754705802942,
    "Li|H": 10.295240079555905,
    "Li|Li": 0.17163539080732038,
    "Li|O": 3.1490611966268243,
    "Li|P": 0.9219579946175536,
    "O|C": 1.648374678701409,
    "O|F": 0.3376246372496941,
    "O|H": 1.741256011906126,
    "O|Li": 0.1056288870985937,
    "O|O": 1.7914538119323113,
    "O|P": 0.2940408372266978,
    "P|C": 29.3667218589119,
    "P|F": 5.453179058857958,  # should be 6
    "P|H": 18.338368341458043,
    "P|Li": 0.9219579946175536,
    "P|O": 8.76609245982093,
    "P|P": 0.2717499493665311,
}

# Reference first shell distances (in Å) for each atom pair
# These represent the distance to the first minimum in the RDF for each pair
# Computed from the same EC-EMC test system and parameters as above
dist = {
    "C|C": 2.75,
    "C|F": 4.55,
    "C|H": 2.45,
    "C|Li": 3.55,
    "C|O": 1.85,
    "C|P": 6.85,
    "F|C": 4.55,
    "F|F": 2.75,
    "F|H": 3.55,
    "F|Li": 2.75,
    "F|O": 3.75,
    "F|P": 4.75,
    "H|C": 2.45,
    "H|F": 3.55,
    "H|H": 2.05,
    "H|Li": 4.35,
    "H|O": 2.35,
    "H|P": 4.85,
    "Li|C": 3.55,
    "Li|F": 2.75,
    "Li|H": 4.35,
    "Li|Li": 5.15,
    "Li|O": 2.85,
    "Li|P": 4.25,
    "O|C": 1.85,
    "O|F": 3.75,
    "O|H": 2.35,
    "O|Li": 2.85,
    "O|O": 2.55,
    "O|P": 5.15,
    "P|C": 6.85,
    "P|F": 4.75,
    "P|H": 4.85,
    "P|Li": 4.25,
    "P|O": 5.15,
    "P|P": 6.05,
}


def test_coordination_number_node(tmp_path, ec_emc):
    """Test the CoordinationNumber node with RDF dependency."""
    os.chdir(tmp_path)

    # First create an RDF node
    rdf_node = massband.RadialDistributionFunction(
        file=ec_emc,
        structures=None,
        stop=1000,  # Use fewer frames for faster testing
        bin_width=0.1,
    )
    rdf_node.run()

    coord_node = massband.CoordinationNumber(
        rdf=rdf_node,
        density_threshold=0.5,
        max_integration_distance=8.0,
    )
    coord_node.run()

    for key in cn:
        assert coord_node.coordination_numbers[key] == pytest.approx(cn[key], rel=1e-5)
        assert coord_node.first_shell_distances[key] == pytest.approx(dist[key], rel=1e-5)


def test_coordination_number_node_substructure_rdf(tmp_path, ec_emc, ec_emc_smiles):
    """Test CoordinationNumber with substructure RDF."""
    os.chdir(tmp_path)

    rdf_node = massband.SubstructureRadialDistributionFunction(
        file=ec_emc,
        structures=ec_emc_smiles,
        pairs=[("[F]", "[P]")],
        hydrogens=[("include", "include")],
        stop=1000,  # Use fewer frames for faster testing
        bin_width=0.1,
    )
    rdf_node.run()

    coord_node = massband.CoordinationNumber(
        rdf=rdf_node,
        density_threshold=0.5,
        max_integration_distance=8.0,
    )
    coord_node.run()

    assert set(coord_node.coordination_numbers) == {"[F]|[P]", "[P]|[F]"}
