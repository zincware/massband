from pathlib import Path

import pytest

EC_EMC = (Path(__file__).parent.parent / "data" / "ec_emc.h5").resolve()


@pytest.fixture
def ec_emc():
    """Fixture to provide the EC EMC file."""
    return EC_EMC


@pytest.fixture
def ec_emc_smiles() -> list[str]:
    smiles = {
        "PF6": "F[P-](F)(F)(F)(F)F",
        "Li": "[Li+]",
        "EC": "C1COC(=O)O1",
        "EMC": "CCOC(=O)OC",
        "VC": "C1=COC(=O)O1",
        "DMC": "COC(=O)OC",
    }
    return list(smiles.values())
