"""Test the refactored RDF computation functions."""

import jax.numpy as jnp
import numpy as np
import pytest

from massband.rdf.core import (
    _apply_cea_reweighting,
    _compute_frame_histograms,
    _normalize_histogram_to_rdf,
    compute_rdf,
)


def test_normalize_histogram_to_rdf():
    """Test histogram normalization function."""
    # Simple test case
    n_bins = 10
    n_frames = 5
    hist = jnp.ones(n_bins)
    n_a, n_b = 10, 10

    # Create simple cubic cells
    cell = jnp.tile(jnp.eye(3) * 10.0, (n_frames, 1, 1))
    bin_edges = jnp.linspace(0, 5, n_bins + 1)

    g_r, dens_a, dens_b = _normalize_histogram_to_rdf(
        hist, n_a, n_b, cell, bin_edges, exclude_self=False
    )

    # Check outputs have correct shape
    assert g_r.shape == (n_bins,)
    assert jnp.ndim(dens_a) == 0  # Scalar
    assert jnp.ndim(dens_b) == 0  # Scalar

    # Check densities are positive
    assert float(dens_a) > 0
    assert float(dens_b) > 0


def test_compute_frame_histograms():
    """Test frame histogram computation."""
    n_frames = 3
    n_particles_a = 5
    n_particles_b = 5

    # Create random positions in a 10x10x10 box
    rng = np.random.default_rng(42)
    positions_a = jnp.array(rng.random((n_frames, n_particles_a, 3)) * 10.0)
    positions_b = jnp.array(rng.random((n_frames, n_particles_b, 3)) * 10.0)
    cell = jnp.tile(jnp.eye(3) * 10.0, (n_frames, 1, 1))

    bin_edges = jnp.linspace(0, 5, 11)

    hist = _compute_frame_histograms(
        positions_a, positions_b, cell, bin_edges, exclude_self=False, batch_size=2
    )

    # Check output shape
    assert hist.shape == (n_frames, len(bin_edges) - 1)

    # Check histograms are non-negative
    assert jnp.all(hist >= 0)


def test_apply_cea_reweighting():
    """Test CEA reweighting function."""
    n_frames = 10
    n_bins = 5

    # Create dummy histogram data
    hist_per_frame = jnp.ones((n_frames, n_bins))

    # Create dummy energy data
    rng = np.random.default_rng(42)
    energy_model = jnp.array(rng.random(n_frames))
    energy_mean = jnp.array(rng.random(n_frames))
    temperature = 300.0

    hist_cea = _apply_cea_reweighting(
        hist_per_frame, energy_model, energy_mean, temperature
    )

    # Check output shape
    assert hist_cea.shape == (n_bins,)


def test_compute_rdf_standard():
    """Test standard RDF computation (no uncertainty)."""
    n_frames = 5
    n_particles_a = 8
    n_particles_b = 8

    # Create random positions
    rng = np.random.default_rng(42)
    positions_a = jnp.array(rng.random((n_frames, n_particles_a, 3)) * 10.0)
    positions_b = jnp.array(rng.random((n_frames, n_particles_b, 3)) * 10.0)
    cell = jnp.tile(jnp.eye(3) * 10.0, (n_frames, 1, 1))

    bin_edges = jnp.linspace(0, 5, 11)

    g_r, g_r_std, g_r_ensemble, dens_a, dens_b = compute_rdf(
        positions_a,
        positions_b,
        cell,
        bin_edges,
        batch_size=2,
        exclude_self=False,
    )

    # Check outputs
    assert g_r.shape == (len(bin_edges) - 1,)
    assert g_r_std is None  # Should be None for standard RDF
    assert g_r_ensemble is None  # Should be None for standard RDF
    assert jnp.ndim(dens_a) == 0  # Scalar
    assert jnp.ndim(dens_b) == 0  # Scalar


def test_compute_rdf_with_ensemble():
    """Test RDF computation with ensemble uncertainty."""
    n_frames = 5
    n_models = 3
    n_particles_a = 8
    n_particles_b = 8

    # Create random positions
    rng = np.random.default_rng(42)
    positions_a = jnp.array(rng.random((n_frames, n_particles_a, 3)) * 10.0)
    positions_b = jnp.array(rng.random((n_frames, n_particles_b, 3)) * 10.0)
    cell = jnp.tile(jnp.eye(3) * 10.0, (n_frames, 1, 1))

    # Create dummy energy ensemble
    energy_ensemble = jnp.array(rng.random((n_frames, n_models)))

    bin_edges = jnp.linspace(0, 5, 11)

    g_r, g_r_std, g_r_ensemble, dens_a, dens_b = compute_rdf(
        positions_a,
        positions_b,
        cell,
        bin_edges,
        batch_size=2,
        exclude_self=False,
        energy_ensemble=energy_ensemble,
        temperature=300.0,
    )

    # Check outputs
    assert g_r.shape == (len(bin_edges) - 1,)
    assert g_r_std is not None
    assert g_r_std.shape == (len(bin_edges) - 1,)
    assert g_r_ensemble is not None
    assert g_r_ensemble.shape == (n_models, len(bin_edges) - 1)
    assert jnp.ndim(dens_a) == 0  # Scalar
    assert jnp.ndim(dens_b) == 0  # Scalar


def test_compute_rdf_ensemble_requires_temperature():
    """Test that ensemble RDF requires temperature parameter."""
    n_frames = 5
    n_models = 3
    n_particles_a = 8
    n_particles_b = 8

    rng = np.random.default_rng(42)
    positions_a = jnp.array(rng.random((n_frames, n_particles_a, 3)) * 10.0)
    positions_b = jnp.array(rng.random((n_frames, n_particles_b, 3)) * 10.0)
    cell = jnp.tile(jnp.eye(3) * 10.0, (n_frames, 1, 1))
    energy_ensemble = jnp.array(rng.random((n_frames, n_models)))
    bin_edges = jnp.linspace(0, 5, 11)

    # Should raise ValueError when temperature is not provided
    with pytest.raises(ValueError, match="temperature must be provided"):
        compute_rdf(
            positions_a,
            positions_b,
            cell,
            bin_edges,
            energy_ensemble=energy_ensemble,
            temperature=None,  # Missing temperature
        )
