"""Test PMF uncertainty propagation."""

import numpy as np

from massband.pmf import PotentialOfMeanForce


def test_pmf_uncertainty_propagation():
    """Test that PMF uncertainty is correctly propagated from RDF uncertainty."""
    # Create a mock PotentialOfMeanForce instance (just for the method)
    pmf = PotentialOfMeanForce.__new__(PotentialOfMeanForce)
    pmf.min_gdr_threshold = 1e-9

    # Create test data
    r = np.linspace(0.1, 5.0, 50)
    g_r = np.exp(-((r - 2.0) ** 2) / 0.5) + 0.5  # Gaussian-like RDF
    g_r_std = 0.1 * np.ones_like(g_r)  # Constant uncertainty
    temperature = 300.0

    # Calculate PMF with uncertainties
    pmf_values, pmf_std = pmf._calculate_pmf(r, g_r, temperature, g_r_std=g_r_std)

    # Check that both are returned
    assert pmf_values is not None
    assert pmf_std is not None
    assert pmf_values.shape == r.shape
    assert pmf_std.shape == r.shape

    # Check that PMF std has reasonable values (not all NaN)
    assert np.sum(np.isfinite(pmf_std)) > 0

    # Check uncertainty propagation formula: δPMF ≈ kT * δg(r) / g(r)
    # where kT ≈ 0.0259 eV at 300K
    kT_300K = 0.0259  # eV (approximately)
    expected_std = kT_300K * g_r_std / g_r

    # Allow for some numerical differences
    valid_mask = np.isfinite(pmf_std)
    np.testing.assert_allclose(pmf_std[valid_mask], expected_std[valid_mask], rtol=0.01)


def test_pmf_without_uncertainty():
    """Test that PMF calculation works without uncertainty."""
    pmf = PotentialOfMeanForce.__new__(PotentialOfMeanForce)
    pmf.min_gdr_threshold = 1e-9

    r = np.linspace(0.1, 5.0, 50)
    g_r = np.exp(-((r - 2.0) ** 2) / 0.5) + 0.5
    temperature = 300.0

    # Calculate PMF without uncertainties
    pmf_values, pmf_std = pmf._calculate_pmf(r, g_r, temperature, g_r_std=None)

    # Check that PMF is returned but std is None
    assert pmf_values is not None
    assert pmf_std is None
    assert pmf_values.shape == r.shape


def test_pmf_handles_zero_rdf():
    """Test that PMF handles zero g(r) values correctly."""
    pmf = PotentialOfMeanForce.__new__(PotentialOfMeanForce)
    pmf.min_gdr_threshold = 1e-9

    r = np.linspace(0.1, 5.0, 50)
    g_r = np.ones_like(r)
    g_r[20:30] = 0  # Set some values to zero
    g_r_std = 0.1 * np.ones_like(g_r)
    temperature = 300.0

    pmf_values, pmf_std = pmf._calculate_pmf(r, g_r, temperature, g_r_std=g_r_std)

    # Check that PMF is NaN where g(r) is zero
    assert np.all(np.isnan(pmf_values[20:30]))

    # Check that PMF_std is also NaN where g(r) is zero
    assert np.all(np.isnan(pmf_std[20:30]))


def test_pmf_normalization_preserves_uncertainty():
    """Test that PMF normalization doesn't affect uncertainty."""
    pmf = PotentialOfMeanForce.__new__(PotentialOfMeanForce)
    pmf.min_gdr_threshold = 1e-9

    r = np.linspace(0.1, 10.0, 100)
    # Create RDF that goes to 1 at large distances
    g_r = 1.0 + np.exp(-((r - 2.0) ** 2) / 0.5)
    g_r_std = 0.1 * np.ones_like(g_r)
    temperature = 300.0

    # Calculate PMF with uncertainty
    pmf_values, pmf_std_before = pmf._calculate_pmf(r, g_r, temperature, g_r_std=g_r_std)

    # Calculate expected uncertainty before normalization
    kT_300K = 0.0259
    expected_std = kT_300K * g_r_std / g_r

    # The normalization shifts PMF by a constant, which shouldn't affect std
    valid_mask = np.isfinite(pmf_std_before) & np.isfinite(expected_std)
    np.testing.assert_allclose(
        pmf_std_before[valid_mask], expected_std[valid_mask], rtol=0.01
    )
