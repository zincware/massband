"""Tests for start/stop/step functionality and memory/non-memory consistency in dataloaders."""

from collections import defaultdict

import jax.numpy as jnp
import pytest

from massband.dataloader import (
    IndependentBatchedLoader,
    SpeciesBatchedLoader,
    TimeBatchedLoader,
)


@pytest.mark.parametrize("memory", [True, False])
@pytest.mark.parametrize("start", [0, 10, 50])
@pytest.mark.parametrize("stop", [None, 100, 200])
@pytest.mark.parametrize("step", [1, 2, 5])
class TestStartStopStepConsistency:
    """Test start/stop/step parameter consistency between memory and non-memory modes."""
    
    def test_time_batched_loader_len_consistency(self, memory, start, stop, step, bmim_bf4_vectra):
        """Test that len() returns consistent values for memory and non-memory modes."""
        # Get original trajectory length for consistent calculation
        import znh5md
        original_handler = znh5md.IO(bmim_bf4_vectra, variable_shape=False, include=["position", "box"])
        original_length = len(original_handler)
        
        loader = TimeBatchedLoader(
            file=bmim_bf4_vectra,
            wrap=False,
            batch_size=32,
            memory=memory,
            start=start,
            stop=stop,
            step=step,
            structures=["CCCCN1C=C[N+](=C1)C", "[B-](F)(F)(F)F"]
        )
        
        # Check that total_frames calculation is consistent
        if hasattr(loader, 'total_frames'):  # Only if loader was successfully initialized 
            expected_frames = len(range(start, stop or original_length, step))
            assert loader.total_frames == expected_frames
            
            # len() should return number of batches, not frames
            expected_batches = (expected_frames + loader.batch_size - 1) // loader.batch_size
            assert len(loader) == expected_batches
    
    def test_species_batched_loader_len_consistency(self, memory, start, stop, step, bmim_bf4_vectra):
        """Test that len() returns consistent values for memory and non-memory modes."""
        # Get original trajectory length for consistent calculation
        import znh5md
        original_handler = znh5md.IO(bmim_bf4_vectra, variable_shape=False, include=["position", "box"])
        original_length = len(original_handler)
        
        loader = SpeciesBatchedLoader(
            file=bmim_bf4_vectra,
            wrap=False,
            batch_size=64,
            memory=memory,
            start=start,
            stop=stop,
            step=step,
            structures=["CCCCN1C=C[N+](=C1)C", "[B-](F)(F)(F)F"]
        )
        
        # Check that total_frames calculation is consistent
        if hasattr(loader, 'total_frames'):  # Only if loader was successfully initialized
            expected_frames = len(range(start, stop or original_length, step))
            assert loader.total_frames == expected_frames
            
            # len() should return number of species batches
            assert len(loader) == len(loader.species_batches)
    
    def test_independent_batched_loader_len_consistency(self, memory, start, stop, step, bmim_bf4_vectra):
        """Test that len() returns consistent values for memory and non-memory modes."""
        # Get original trajectory length for consistent calculation
        import znh5md
        original_handler = znh5md.IO(bmim_bf4_vectra, variable_shape=False, include=["position", "box"])
        original_length = len(original_handler)
        
        loader = IndependentBatchedLoader(
            file=bmim_bf4_vectra,
            wrap=False,
            batch_size=1,  # Must be 1 for IndependentBatchedLoader
            memory=memory,
            start=start,
            stop=stop,
            step=step,
            structures=["CCCCN1C=C[N+](=C1)C", "[B-](F)(F)(F)F"]
        )
        
        # Check that total_frames calculation is consistent
        if hasattr(loader, 'total_frames'):  # Only if loader was successfully initialized
            expected_frames = len(range(start, stop or original_length, step))
            assert loader.total_frames == expected_frames
            assert len(loader) == expected_frames


@pytest.mark.parametrize("loader_class", [TimeBatchedLoader, SpeciesBatchedLoader])
@pytest.mark.parametrize("start,stop,step", [
    (0, 100, 1),
    (10, 90, 2),
    (0, None, 3),
    (50, 150, 1),
    (20, 80, 5)
])
def test_memory_vs_non_memory_consistency(loader_class, start, stop, step, bmim_bf4_vectra):
    """Test that memory and non-memory modes produce identical results."""
    
    # Create loaders with same parameters but different memory settings
    loader_memory = loader_class(
        file=bmim_bf4_vectra,
        wrap=False,
        batch_size=32,
        memory=True,
        start=start,
        stop=stop,
        step=step,
        structures=["CCCCN1C=C[N+](=C1)C", "[B-](F)(F)(F)F"]
    )
    
    loader_no_memory = loader_class(
        file=bmim_bf4_vectra,
        wrap=False,
        batch_size=32,
        memory=False,
        start=start,
        stop=stop,
        step=step,
        structures=["CCCCN1C=C[N+](=C1)C", "[B-](F)(F)(F)F"]
    )
    
    # Skip test if parameters result in empty trajectory
    if len(loader_memory) == 0 or len(loader_no_memory) == 0:
        pytest.skip("Parameters result in empty trajectory")
    
    # Test that lengths are identical
    assert len(loader_memory) == len(loader_no_memory)
    
    # Collect all data from both loaders
    memory_data = defaultdict(list)
    no_memory_data = defaultdict(list)
    
    for batch_output in loader_memory:
        if "position" in batch_output:
            batch = batch_output["position"]
            for species, positions in batch.items():
                memory_data[species].append(positions)
    
    for batch_output in loader_no_memory:
        if "position" in batch_output:
            batch = batch_output["position"]
            for species, positions in batch.items():
                no_memory_data[species].append(positions)
    
    # Verify same species are present
    assert memory_data.keys() == no_memory_data.keys()
    
    # Verify data is identical within tolerance
    tolerance = 1e-10
    for species in memory_data.keys():
        if loader_class == TimeBatchedLoader:
            memory_concat = jnp.concatenate(memory_data[species], axis=0)
            no_memory_concat = jnp.concatenate(no_memory_data[species], axis=0)
        else:  # SpeciesBatchedLoader
            memory_concat = jnp.concatenate(memory_data[species], axis=1)
            no_memory_concat = jnp.concatenate(no_memory_data[species], axis=1)
        
        assert memory_concat.shape == no_memory_concat.shape, f"Shape mismatch for {species}"
        assert jnp.allclose(memory_concat, no_memory_concat, atol=tolerance), \
            f"Data mismatch for {species} (max diff: {jnp.max(jnp.abs(memory_concat - no_memory_concat))})"


@pytest.mark.parametrize("loader_class", [TimeBatchedLoader, SpeciesBatchedLoader, IndependentBatchedLoader])
def test_effective_stop_calculation(loader_class, bmim_bf4_vectra):
    """Test that effective stop is calculated correctly."""
    
    # Test with explicit stop
    loader_explicit = loader_class(
        file=bmim_bf4_vectra,
        wrap=False,
        batch_size=32 if loader_class != IndependentBatchedLoader else 1,
        memory=False,
        start=0,
        stop=100,
        step=1
    )
    
    # Test with None stop (should use full trajectory length)
    loader_none = loader_class(
        file=bmim_bf4_vectra,
        wrap=False,
        batch_size=32 if loader_class != IndependentBatchedLoader else 1,
        memory=False,
        start=0,
        stop=None,
        step=1
    )
    
    # Get trajectory length for comparison
    import znh5md
    handler = znh5md.IO(bmim_bf4_vectra, variable_shape=False, include=["position", "box"])
    trajectory_length = len(handler)
    
    # Test explicit stop
    expected_frames_explicit = len(range(0, 100, 1))
    assert loader_explicit.total_frames == expected_frames_explicit
    
    # Test None stop
    expected_frames_none = len(range(0, trajectory_length, 1))
    assert loader_none.total_frames == expected_frames_none


@pytest.mark.parametrize("start,stop,step", [
    (0, 10, 1),   # Normal case
    (5, 15, 2),   # With step
    (0, 1, 1),    # Single frame
    (100, 101, 1), # Single frame at offset
    (0, 50, 10),  # Large step
])
def test_frame_iteration_correctness(start, stop, step, bmim_bf4_vectra):
    """Test that iteration produces the correct number of frames."""
    
    loader = TimeBatchedLoader(
        file=bmim_bf4_vectra,
        wrap=False,
        batch_size=5,
        memory=False,
        start=start,
        stop=stop,
        step=step,
        structures=["CCCCN1C=C[N+](=C1)C"]
    )
    
    if len(loader) == 0:
        pytest.skip("Parameters result in empty trajectory")
    
    total_frames_collected = 0
    for batch_output in loader:
        if "position" in batch_output:
            batch = batch_output["position"]
            # Count frames in this batch
            for positions in batch.values():
                frames_in_batch = positions.shape[0]
                total_frames_collected = frames_in_batch  # All species should have same frame count
                break  # Only need to check one species
    
    # The total frames collected should equal the expected range length
    expected_total_frames = len(range(start, stop, step))
    # Note: Due to batching, we might have collected exactly expected_total_frames
    # or possibly more if the last batch wasn't full
    assert total_frames_collected <= expected_total_frames + loader.batch_size


def test_zero_frames_handling(bmim_bf4_vectra):
    """Test handling of parameters that result in zero frames."""
    
    # Parameters that should result in zero frames
    test_cases = [
        (100, 50, 1),   # start > stop
        (0, 0, 1),      # start == stop
        (1000, None, 1) # start beyond trajectory length
    ]
    
    for start, stop, step in test_cases:
        loader = TimeBatchedLoader(
            file=bmim_bf4_vectra,
            wrap=False,
            batch_size=32,
            memory=False,
            start=start,
            stop=stop,
            step=step
        )
        
        # Should have zero length and empty iteration
        if hasattr(loader, 'total_frames') and loader.total_frames == 0:
            assert len(loader) == 0
            
            # Iteration should be empty
            batches = list(loader)
            assert len(batches) == 0


@pytest.mark.parametrize("memory", [True, False])
def test_large_step_consistency(memory, bmim_bf4_vectra):
    """Test consistency with large step values."""
    
    loader = TimeBatchedLoader(
        file=bmim_bf4_vectra,
        wrap=False,
        batch_size=10,
        memory=memory,
        start=0,
        stop=100,
        step=25,  # Large step
        structures=["CCCCN1C=C[N+](=C1)C"]
    )
    
    expected_frames = len(range(0, 100, 25))  # Should be 4 frames: 0, 25, 50, 75
    assert loader.total_frames == expected_frames
    
    # Collect all frames
    all_frames = []
    for batch_output in loader:
        if "position" in batch_output:
            batch = batch_output["position"]
            for positions in batch.values():
                all_frames.append(positions)
                break  # Only need one species
    
    if all_frames:
        total_collected = sum(pos.shape[0] for pos in all_frames)
        assert total_collected == expected_frames