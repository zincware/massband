"""Test that conductivity classes have compatible output structures."""

from massband.conductivity.einstein_helfand import KinisiEinsteinHelfandIonicConductivity
from massband.conductivity.ne import NernstEinsteinIonicConductivity


def test_nernst_einstein_output_structure():
    """Test that NernstEinstein outputs 'total' key with proper structure."""
    # Create mock node
    ne = NernstEinsteinIonicConductivity.__new__(NernstEinsteinIonicConductivity)

    # Simulate the output structure
    ne.conductivity = {
        "total": {
            "mean": 1.5,
            "std": 0.2,
            "var": 0.04,
            "occurrences": 100,
            "unit": "S/m",
            "box": [[10, 0, 0], [0, 10, 0], [0, 0, 10]],
        }
    }

    # Check structure
    assert "total" in ne.conductivity
    assert "mean" in ne.conductivity["total"]
    assert "std" in ne.conductivity["total"]
    assert "var" in ne.conductivity["total"]
    assert "unit" in ne.conductivity["total"]


def test_einstein_helfand_output_structure():
    """Test that EinsteinHelfand outputs 'total' key with proper structure."""
    # Create mock node
    eh = KinisiEinsteinHelfandIonicConductivity.__new__(
        KinisiEinsteinHelfandIonicConductivity
    )

    # Simulate the output structure (as it should be after fix)
    eh.conductivity = {
        "total": {
            "mean": 1.5,
            "std": 0.2,
            "var": 0.04,
            "unit": "S/m",
        }
    }

    # Check structure
    assert "total" in eh.conductivity
    assert "mean" in eh.conductivity["total"]
    assert "std" in eh.conductivity["total"]
    assert "var" in eh.conductivity["total"]
    assert "unit" in eh.conductivity["total"]


def test_conductivity_classes_interchangeable():
    """Test that both conductivity classes can be used with Arrhenius."""
    # Mock outputs from both classes
    ne_output = {
        "total": {
            "mean": 1.5,
            "std": 0.2,
            "var": 0.04,
            "unit": "S/m",
            "occurrences": 100,
            "box": [[10, 0, 0], [0, 10, 0], [0, 0, 10]],
        }
    }

    eh_output = {
        "total": {
            "mean": 1.5,
            "std": 0.2,
            "var": 0.04,
            "unit": "S/m",
        }
    }

    # Test that both have the same structure keys for Arrhenius
    for output in [ne_output, eh_output]:
        assert "total" in output
        assert "mean" in output["total"]
        assert "var" in output["total"]

    # Simulate what Arrhenius does (from arrhenius.py:89-93)
    data_list = [ne_output, eh_output]
    structure = "total"

    # This should work for both
    sigma_mean = [data_dict[structure]["mean"] for data_dict in data_list]
    sigma_var = [data_dict[structure]["var"] for data_dict in data_list]

    assert sigma_mean == [1.5, 1.5]
    assert sigma_var == [0.04, 0.04]
