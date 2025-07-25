import os

from massband.pmf import PotentialOfMeanForce
from massband.rdf import RadialDistributionFunction


def test_pmf_node(tmp_path, ec_emc, ec_emc_smiles):
    """Test the PotentialOfMeanForce node with RDF dependency."""
    os.chdir(tmp_path)

    # First create an RDF node
    rdf_node = RadialDistributionFunction(
        file=ec_emc,
        structures=ec_emc_smiles,
        stop=100,  # Use fewer frames for faster testing
        bin_width=0.1,
    )
    rdf_node.run()

    # Create PMF node
    pmf_node = PotentialOfMeanForce(
        rdf=rdf_node,
        temperature=300.0,
    )
    pmf_node.run()

    assert set(pmf_node.pmf_values.keys()) == set(rdf_node.results.keys())
