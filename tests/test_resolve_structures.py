"""Tests for resolve_structures utility function."""

import ase
import numpy as np
import pytest

from massband.utils import resolve_structures


@pytest.fixture
def water_frame():
    """A simple water-like frame: O H H O H H (two water molecules)."""
    atoms = ase.Atoms(
        symbols="OHHOHH",
        positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [-0.24, 0.93, 0.0],
            [3.0, 0.0, 0.0],
            [3.96, 0.0, 0.0],
            [2.76, 0.93, 0.0],
        ],
    )
    return atoms


def test_resolve_structures_none(water_frame):
    """None groups by element symbol."""
    molecules, masses = resolve_structures(water_frame, None)
    assert set(molecules.keys()) == {"O", "H"}
    # Each atom is a single-atom molecule
    assert len(molecules["O"]) == 2
    assert len(molecules["H"]) == 4
    # Each molecule instance is a 1-tuple
    assert all(len(m) == 1 for m in molecules["O"])
    assert all(len(m) == 1 for m in molecules["H"])
    # Masses should be a list with one element per molecule instance atom
    assert len(masses["O"]) == 1
    assert len(masses["H"]) == 1
    assert masses["O"][0] == pytest.approx(15.999, abs=0.01)
    assert masses["H"][0] == pytest.approx(1.008, abs=0.01)


def test_resolve_structures_dict(water_frame):
    """Dict maps names to flat atom-index lists."""
    structures = {
        "oxygen": [0, 3],
        "hydrogen": [1, 2, 4, 5],
    }
    molecules, masses = resolve_structures(water_frame, structures)
    assert set(molecules.keys()) == {"oxygen", "hydrogen"}
    # Each index becomes a single-atom molecule
    assert molecules["oxygen"] == ((0,), (3,))
    assert molecules["hydrogen"] == ((1,), (2,), (4,), (5,))
    # Masses from first index
    assert masses["oxygen"] == [pytest.approx(15.999, abs=0.01)]
    assert masses["hydrogen"] == [pytest.approx(1.008, abs=0.01)]


def test_resolve_structures_dict_empty_indices(water_frame):
    """Dict with empty index list produces empty molecules and masses."""
    structures = {"empty": []}
    molecules, masses = resolve_structures(water_frame, structures)
    assert molecules["empty"] == ()
    assert masses["empty"] == []


def test_resolve_structures_list_no_match():
    """List with SMILES that don't match produces no entries for that species."""
    atoms = ase.Atoms("Ar", positions=[[0, 0, 0]])
    molecules, masses = resolve_structures(atoms, ["[Li+]"])
    assert "Li+" not in molecules or len(molecules.get("[Li+]", ())) == 0
