"""Test PMF with new dict[str, RDFData] interface."""

import numpy as np

from massband.pmf import PotentialOfMeanForce


def test_pmf_with_rdf_dict():
    """Test PMF can be created directly from RDF data dictionary."""
    # Create mock RDF data
    bin_centers = np.linspace(0.1, 5.0, 50).tolist()
    g_r = (np.exp(-((np.array(bin_centers) - 2.0) ** 2) / 0.5) + 0.5).tolist()

    rdf_data = {
        "Li-Li": {
            "bin_centers": bin_centers,
            "g_r": g_r,
            "g_r_std": None,
            "g_r_ensemble": None,
            "unit": "angstrom",
            "number_density_a": 0.01,
            "number_density_b": 0.01,
        }
    }

    # Create PMF node directly with data dict
    pmf = PotentialOfMeanForce.__new__(PotentialOfMeanForce)
    pmf.data = rdf_data
    pmf.temperature = 300.0
    pmf.min_gdr_threshold = 1e-9
    pmf.pmf = {}
    pmf.figures = None  # Won't create plots in this test

    # Run calculation (skip plotting by patching)
    pmf._calculate_pmf(
        np.array(bin_centers), np.array(g_r), pmf.temperature, g_r_std=None
    )

    # Should not raise any errors


def test_pmf_with_uncertainties_dict():
    """Test PMF works with uncertainty data."""
    bin_centers = np.linspace(0.1, 5.0, 50).tolist()
    g_r = (np.exp(-((np.array(bin_centers) - 2.0) ** 2) / 0.5) + 0.5).tolist()
    g_r_std = (0.1 * np.ones(50)).tolist()

    rdf_data = {
        "Li-F": {
            "bin_centers": bin_centers,
            "g_r": g_r,
            "g_r_std": g_r_std,
            "g_r_ensemble": None,
            "unit": "angstrom",
            "number_density_a": 0.01,
            "number_density_b": 0.02,
        }
    }

    # Create PMF node with uncertainties
    pmf = PotentialOfMeanForce.__new__(PotentialOfMeanForce)
    pmf.data = rdf_data
    pmf.temperature = 300.0
    pmf.min_gdr_threshold = 1e-9

    # Calculate PMF with uncertainty
    pmf_values, pmf_std = pmf._calculate_pmf(
        np.array(bin_centers),
        np.array(g_r),
        pmf.temperature,
        g_r_std=np.array(g_r_std),
    )

    # Check that uncertainties are computed
    assert pmf_std is not None
    assert np.sum(np.isfinite(pmf_std)) > 0
