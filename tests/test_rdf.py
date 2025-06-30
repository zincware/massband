import os
from pathlib import Path

from massband.rdf import RadialDistributionFunction

BMIM_BF4_FILE = (Path(__file__).parent.parent / "data" / "bmim_bf4.h5").resolve()


def test_rdf_node_smiles(tmp_path):
    os.chdir(tmp_path)
    node = RadialDistributionFunction(
        file=BMIM_BF4_FILE,
        batch_size=5,
        structures=["CCCCN1C=C[N+](=C1)C", "[B-](F)(F)(F)F"],
    )
    node.run()
    assert len(node.results[("CCCCN1C=C[N+](=C1)C", "[B-](F)(F)(F)F")]) == 200
    assert len(node.results[("[B-](F)(F)(F)F", "[B-](F)(F)(F)F")]) == 200
    assert len(node.results[("CCCCN1C=C[N+](=C1)C", "CCCCN1C=C[N+](=C1)C")]) == 200
    assert len(node.results) == 3

    assert (node.figures / "rdf.png").exists()
