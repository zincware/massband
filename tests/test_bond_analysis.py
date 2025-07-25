import pytest
from rdkit import Chem

from massband.rdf import select_atoms_flat_unique


@pytest.fixture
def ethanol_mol():
    """Returns an RDKit molecule for ethanol (CCO) with explicit hydrogens."""
    mol = Chem.MolFromSmiles("CCO")
    return Chem.AddHs(mol)


@pytest.fixture
def toluene_mol():
    """Returns an RDKit molecule for toluene (Cc1ccccc1) with explicit hydrogens."""
    mol = Chem.MolFromSmiles("Cc1ccccc1")
    return Chem.AddHs(mol)


# =============================================================================
# Test Cases
# =============================================================================


def test_select_carbons(ethanol_mol):
    """Test selecting all carbon atoms."""
    # Ethanol (CCO with Hs): C(0), C(1), O(2), H(3-8)
    indices = select_atoms_flat_unique(ethanol_mol, "[#6]")
    assert sorted(indices) == [0, 1]


def test_select_oxygen(ethanol_mol):
    """Test selecting the oxygen atom."""
    indices = select_atoms_flat_unique(ethanol_mol, "[#8]")
    assert sorted(indices) == [2]


def test_no_matches(ethanol_mol):
    """Test a SMARTS pattern that has no matches."""
    indices = select_atoms_flat_unique(ethanol_mol, "[F]")  # Fluorine
    assert indices == []


# --- Hydrogen Handling Tests ---


def test_hydrogens_excluded_by_default(ethanol_mol):
    """Test that hydrogens are excluded by default."""
    # C-O bond involves atoms 1 and 2. Hydrogens attached are not included.
    indices = select_atoms_flat_unique(ethanol_mol, "CO")
    assert sorted(indices) == [1, 2]


def test_hydrogens_included(ethanol_mol):
    """Test the 'include' option for hydrogens."""
    # C-O bond (atoms 1, 2) plus attached hydrogens (atoms 6, 7, 8)
    # H on O is atom 8. Hs on C(1) are 6, 7.
    indices = select_atoms_flat_unique(ethanol_mol, "CO", hydrogens="include")
    # Expected: C(1), O(2), H(6), H(7), H(8)
    assert sorted(indices) == [1, 2, 6, 7, 8]


def test_hydrogens_isolated(ethanol_mol):
    """Test the 'isolated' option for hydrogens."""
    # Select ONLY the hydrogens from the C-O match
    indices = select_atoms_flat_unique(ethanol_mol, "CO", hydrogens="isolated")
    # Expected: H(6), H(7), H(8)
    assert sorted(indices) == [6, 7, 8]


def test_smarts_with_explicit_hydrogens(ethanol_mol):
    """Test a SMARTS pattern that explicitly includes hydrogens."""
    # Find all hydrogens attached to an oxygen
    indices = select_atoms_flat_unique(ethanol_mol, "[#8]-[H]", hydrogens="include")
    # Expected: O(2), H(8)
    assert sorted(indices) == [2, 8]

    # Now isolate only the hydrogen from that match
    h_indices = select_atoms_flat_unique(ethanol_mol, "[#8]-[H]", hydrogens="isolated")
    assert sorted(h_indices) == [8]


# --- Mapped SMILES Tests ---


def test_mapped_smiles(ethanol_mol):
    """Test selecting only mapped atoms using a mapped SMILES pattern."""
    # The pattern "[C:1][C:2]O" matches atoms 0, 1, and 2, but only C:1 and C:2 are mapped.
    # The function should now return only the indices of the mapped atoms.
    indices = select_atoms_flat_unique(ethanol_mol, "[C:1][C:2]O")
    # FIX: The test's expectation is updated to only expect the mapped carbons [0, 1].
    assert sorted(indices) == [0, 1]


def test_mapped_smiles_with_hydrogens(ethanol_mol):
    """Test mapped SMILES with hydrogen filtering."""
    # Pattern "C[O:1]" matches atoms C(1) and O(2), but only O(2) is mapped.
    # The core selection will be just atom 2.

    # Include hydrogens attached to the mapped oxygen
    indices_included = select_atoms_flat_unique(
        ethanol_mol, "C[O:1]", hydrogens="include"
    )
    # Expected: O(2) and its hydrogen H(8)
    assert sorted(indices_included) == [2, 8]

    # Exclude hydrogens (returns just the mapped heavy atom)
    indices_excluded = select_atoms_flat_unique(
        ethanol_mol, "C[O:1]", hydrogens="exclude"
    )
    assert sorted(indices_excluded) == [2]

    # Isolate only hydrogens attached to the mapped oxygen
    indices_isolated = select_atoms_flat_unique(
        ethanol_mol, "C[O:1]", hydrogens="isolated"
    )
    assert sorted(indices_isolated) == [8]


# --- Error Handling Tests ---


def test_invalid_smarts_raises_error():
    """Test that an invalid SMARTS string raises a ValueError."""
    mol = Chem.MolFromSmiles("C")
    with pytest.raises(ValueError, match="Invalid SMARTS/SMILES"):
        select_atoms_flat_unique(mol, "this is not valid")
