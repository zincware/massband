import os
from pathlib import Path

from massband.rdf import RadialDistributionFunction

BMIM_BF4_FILE = (Path(__file__).parent.parent / "data" / "bmim_bf4.h5").resolve()


def test_rdf_node(tmp_path):
    os.chdir(tmp_path)
    node = RadialDistributionFunction(
        file=BMIM_BF4_FILE,
        batch_size=5,
        structures=["CCCCN1C=C[N+](=C1)C", "[B-](F)(F)(F)F"],
    )
    node.run()

    # Check if the output file is created in the node's output directory
    assert (node.figures / "rdf.png").exists()
