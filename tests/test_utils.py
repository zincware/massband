"""Tests for utility functions."""

from massband.utils import sanitize_structure_name


def test_short_smiles_clean():
    """Short smiles are sanitized without hash suffix."""
    result = sanitize_structure_name("CCO")
    assert result == "CCO"


def test_long_smiles_clean_truncation():
    """Long smiles are truncated and get a hash suffix."""
    long_smiles = "C" * 100  # 100 character SMILES
    result = sanitize_structure_name(long_smiles)
    # default max_length is 8 (after truncation, but without hash)
    assert len(result) == 8 + 9  # 8 chars + '_' + 8 char hash
    assert result.startswith("CCCCCCCC_")


def test_short_smiles_sanitize():
    """Short smiles with unsafe chars are sanitized and get a hash suffix."""
    result = sanitize_structure_name("C(C)C")
    assert result.startswith("C_C_C")
    assert len(result) == 14  # "C_C_C" (5) + "_" (1) + 8 char hash (8) = 14
    result = sanitize_structure_name("C#C")
    assert result.startswith("C_C")
    assert len(result) == 12  # "C_C" (3) + "_" (1) + 8 char hash (8) = 12


def test_unsafe_filesystem_chars():
    """Test that all unsafe filesystem characters are removed."""
    unsafe_smiles = 'test/\\|:*?"<>[]{}()+=!@#$%^&-test'
    result = sanitize_structure_name(unsafe_smiles)
    # None of the unsafe chars should be in the result (except underscore)
    for char in '/\\|:*?"<>()[]{}+=!@#$%^&-':
        assert char not in result


def test_collision_prevention():
    """Test that different inputs with same sanitized form get unique outputs."""
    # These structures sanitize to the same form but should have different hashes
    result1 = sanitize_structure_name("C-C")  # hyphen
    result2 = sanitize_structure_name("C|C")  # pipe
    result3 = sanitize_structure_name("C(C)")  # parentheses

    # All sanitize to "C_C" but should have different hashes
    assert result1.startswith("C_C_")
    assert result2.startswith("C_C_")
    assert result3.startswith("C_C_")
    assert result1 != result2  # Different hashes
    assert result2 != result3  # Different hashes
    assert result1 != result3  # Different hashes


def test_empty_string():
    """Test handling of empty string."""
    result = sanitize_structure_name("")
    assert result == ""  # Empty clean string stays empty


def test_only_unsafe_chars():
    """Test string with only unsafe characters."""
    result = sanitize_structure_name("()")
    # Should become "__" + "_" + hash
    assert result.startswith("___")
    assert len(result) == 11  # "__" (2) + "_" (1) + hash (8)


def test_spaces_handled():
    """Test that spaces are replaced with underscores."""
    result = sanitize_structure_name("C C C")
    # Has spaces, so needs hash
    assert result.startswith("C_C_C")
    assert len(result) == 14  # "C_C_C" (5) + "_" (1) + hash (8)


def test_boundary_at_max_length():
    """Test behavior at exactly max_length boundary."""
    # String of exactly max_length (8) that's clean
    result = sanitize_structure_name("CCCCCCCC")
    assert result == "CCCCCCCC"
    assert len(result) == 8

    # String of max_length + 1 that's clean - needs truncation and hash
    result = sanitize_structure_name("CCCCCCCCC")
    assert result.startswith("CCCCCCCC_")
    assert len(result) == 17  # 8 + "_" + 8


def test_custom_max_length():
    """Test with custom max_length parameter."""
    result = sanitize_structure_name("C" * 20, max_length=16)
    # Should truncate to 16 and add hash
    assert len(result) == 25  # 16 + "_" + 8
    assert result.startswith("CCCCCCCCCCCCCCCC_")


def test_deterministic():
    """Test that function is deterministic."""
    structure = "c1ccccc1"
    results = [sanitize_structure_name(structure) for _ in range(5)]
    assert all(r == results[0] for r in results)


def test_pair_names():
    """Test sanitization of pair names like 'Li-TFSI'."""
    result = sanitize_structure_name("Li-TFSI")
    # Has hyphen, so needs sanitization and hash
    assert result.startswith("Li_TFSI")
    assert "_" in result  # Has underscores
    assert len(result) == 16  # "Li_TFSI" (7) + "_" (1) + hash (8)
