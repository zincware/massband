import os

import numpy as np
import pytest

from massband.pmf import PotentialOfMeanForce
from massband.rdf import RadialDistributionFunction


def test_pmf_calculation():
    """Test basic PMF calculation from known g(r) values."""
    # Create a mock RDF node for testing
    class MockRDF:
        def __init__(self):
            self.results = {}
            self.bin_width = 0.1
    
    mock_rdf = MockRDF()
    pmf_calc = PotentialOfMeanForce(rdf=mock_rdf, temperature=300.0, min_gdr_threshold=1e-6)
    
    # Test with known g(r) values
    g_r = np.array([0.0, 0.1, 0.5, 1.0, 2.0, 1.5, 1.0])
    temperature = 300.0
    
    pmf = pmf_calc._calculate_pmf(g_r, temperature)
    
    # Check that PMF = -kT * ln(g(r))
    k_B = 0.001987204  # kcal/(mol·K)
    expected_pmf = -k_B * temperature * np.log(np.maximum(g_r, pmf_calc.min_gdr_threshold))
    
    # Where original g(r) was below threshold, PMF should be NaN
    expected_pmf[g_r < pmf_calc.min_gdr_threshold] = np.nan
    
    # Compare finite values
    finite_mask = np.isfinite(pmf) & np.isfinite(expected_pmf)
    np.testing.assert_array_almost_equal(pmf[finite_mask], expected_pmf[finite_mask])
    
    # Check that g(r) = 1 gives PMF = 0
    assert pmf[3] == pytest.approx(0.0, abs=1e-10)
    assert pmf[6] == pytest.approx(0.0, abs=1e-10)


def test_pmf_node(tmp_path, ec_emc):
    """Test the PotentialOfMeanForce node with RDF dependency."""
    os.chdir(tmp_path)

    # First create an RDF node
    rdf_node = RadialDistributionFunction(
        file=ec_emc,
        structures=None,
        stop=100,  # Use fewer frames for faster testing
        bin_width=0.1,
    )
    rdf_node.run()

    # Create PMF node
    pmf_node = PotentialOfMeanForce(
        rdf=rdf_node,
        temperature=300.0,
    )
    pmf_node.run()

    # Check that PMF values were calculated for all RDF pairs
    assert len(pmf_node.pmf_values) > 0
    assert set(pmf_node.pmf_values.keys()) == set(rdf_node.results.keys())

    # Check specific properties of PMF calculation
    for pair_key, pmf_list in pmf_node.pmf_values.items():
        pmf_array = np.array(pmf_list)
        g_r_array = np.array(rdf_node.results[pair_key])
        
        # Check that arrays have same length
        assert len(pmf_array) == len(g_r_array)
        
        # Where g(r) = 1, PMF should be approximately 0
        unity_mask = np.abs(g_r_array - 1.0) < 1e-6
        if np.any(unity_mask):
            assert np.all(np.abs(pmf_array[unity_mask]) < 1e-6)
        
        # Check that finite PMF values exist
        finite_count = np.sum(np.isfinite(pmf_array))
        assert finite_count > 0, f"No finite PMF values for {pair_key}"


def test_pmf_temperature_dependence():
    """Test that PMF scales correctly with temperature."""
    # Create a mock RDF node for testing
    class MockRDF:
        def __init__(self):
            self.results = {}
            self.bin_width = 0.1
    
    mock_rdf = MockRDF()
    pmf_calc = PotentialOfMeanForce(rdf=mock_rdf, temperature=300.0, min_gdr_threshold=1e-6)
    
    g_r = np.array([0.1, 0.5, 2.0, 1.5])
    
    pmf_300 = pmf_calc._calculate_pmf(g_r, 300.0)
    pmf_600 = pmf_calc._calculate_pmf(g_r, 600.0)
    
    # PMF should scale linearly with temperature
    finite_mask = np.isfinite(pmf_300) & np.isfinite(pmf_600)
    ratio = pmf_600[finite_mask] / pmf_300[finite_mask]
    expected_ratio = 600.0 / 300.0
    
    np.testing.assert_array_almost_equal(ratio, expected_ratio, decimal=10)


def test_pmf_edge_cases():
    """Test PMF calculation with edge cases."""
    # Create a mock RDF node for testing
    class MockRDF:
        def __init__(self):
            self.results = {}
            self.bin_width = 0.1
    
    mock_rdf = MockRDF()
    pmf_calc = PotentialOfMeanForce(rdf=mock_rdf, temperature=300.0, min_gdr_threshold=1e-6)
    
    # Test with all zeros (should give NaN)
    g_r_zeros = np.zeros(5)
    pmf_zeros = pmf_calc._calculate_pmf(g_r_zeros, 300.0)
    assert np.all(np.isnan(pmf_zeros))
    
    # Test with very small values
    g_r_small = np.array([1e-8, 1e-7, 1e-6, 1e-5])
    pmf_small = pmf_calc._calculate_pmf(g_r_small, 300.0)
    
    # Values below threshold should be NaN
    below_threshold = g_r_small < pmf_calc.min_gdr_threshold
    assert np.all(np.isnan(pmf_small[below_threshold]))
    
    # Values above threshold should be finite
    above_threshold = g_r_small >= pmf_calc.min_gdr_threshold
    if np.any(above_threshold):
        assert np.all(np.isfinite(pmf_small[above_threshold]))


def test_pmf_physical_meaning():
    """Test that PMF values have correct physical interpretation."""
    # Create a mock RDF node for testing
    class MockRDF:
        def __init__(self):
            self.results = {}
            self.bin_width = 0.1
    
    mock_rdf = MockRDF()
    pmf_calc = PotentialOfMeanForce(rdf=mock_rdf, temperature=300.0, min_gdr_threshold=1e-6)
    
    # g(r) > 1 should give negative PMF (attractive interaction)
    g_r_attractive = np.array([1.5, 2.0, 3.0])
    pmf_attractive = pmf_calc._calculate_pmf(g_r_attractive, 300.0)
    assert np.all(pmf_attractive < 0)
    
    # g(r) < 1 should give positive PMF (repulsive interaction)
    g_r_repulsive = np.array([0.1, 0.5, 0.8])
    pmf_repulsive = pmf_calc._calculate_pmf(g_r_repulsive, 300.0)
    assert np.all(pmf_repulsive > 0)
    
    # g(r) = 1 should give PMF = 0 (no interaction)
    g_r_neutral = np.array([1.0])
    pmf_neutral = pmf_calc._calculate_pmf(g_r_neutral, 300.0)
    assert np.abs(pmf_neutral[0]) < 1e-10