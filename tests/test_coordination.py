import os

import pytest

from massband.coordination import CoordinationNumber
from massband.rdf import RadialDistributionFunction

# Reference coordination numbers calculated from EC-EMC test system
# These values were computed using the coordination number analysis with:
# - density_threshold=0.5, max_integration_distance=8.0, bin_width=0.1
# - First 1000 frames of the EC-EMC trajectory
# Values represent the coordination number for each atom pair's first shell
cn = {
    "C|C": 2.511757715067251,
    "C|F": 9.562391799136192,
    "C|O": 2.413154668947617,
    "C|P": 42.991707456552255,
    "F|F": 32.878353782740334,
    "F|P": 43.325688457368756,
    "H|C": 2.1663340341806827,
    "H|F": 4.122742313725539,
    "H|H": 1.4489031339535374,
    "H|Li": 8.591267273809493,
    "H|O": 1.4530594405583235,
    "H|P": 15.303171423840036,
    "Li|C": 5.575186007839031,
    "Li|F": 10.358539348706266,
    "Li|Li": 8.181893229745556,
    "Li|O": 5.03535006476462,
    "Li|P": 43.94992104361161,
    "O|F": 2.682438939035287,
    "O|O": 2.8645353344033992,
    "O|P": 14.016985215331388,
    "P|P": 12.954374155862542,
}

# Reference first shell distances (in Å) for each atom pair
# These represent the distance to the first minimum in the RDF for each pair
# Computed from the same EC-EMC test system and parameters as above
dist = {
    "C|C": 2.75,
    "C|F": 4.55,
    "C|O": 1.85,
    "C|P": 6.85,
    "F|F": 2.75,
    "F|P": 4.75,
    "H|C": 2.45,
    "H|F": 3.55,
    "H|H": 2.05,
    "H|Li": 4.35,
    "H|O": 2.35,
    "H|P": 4.85,
    "Li|C": 3.55,
    "Li|F": 2.75,
    "Li|Li": 5.15,
    "Li|O": 2.85,
    "Li|P": 4.25,
    "O|F": 3.75,
    "O|O": 2.55,
    "O|P": 5.15,
    "P|P": 6.05,
}


def test_coordination_number_node(tmp_path, ec_emc):
    """Test the CoordinationNumber node with RDF dependency."""
    os.chdir(tmp_path)

    # First create an RDF node
    rdf_node = RadialDistributionFunction(
        file=ec_emc,
        structures=None,
        stop=1000,  # Use fewer frames for faster testing
        bin_width=0.1,
    )
    rdf_node.run()

    # Create coordination number node that depends on RDF
    coord_node = CoordinationNumber(
        rdf=rdf_node,
        density_threshold=0.5,
        max_integration_distance=8.0,
    )
    coord_node.run()

    for key in cn:
        assert coord_node.coordination_numbers[key] == pytest.approx(cn[key], rel=1e-5)
        assert coord_node.first_shell_distances[key] == pytest.approx(dist[key], rel=1e-5)
