import pytest
from pathlib import Path

BMIM_BF4_VECTRA = (Path(__file__).parent.parent / "data" / "bmim_bf4_vectra.h5").resolve()

@pytest.fixture
def bmim_bf4_vectra():
    """Fixture to provide the BMIM BF4 Vectra file."""
    return BMIM_BF4_VECTRA