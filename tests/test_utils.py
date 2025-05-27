import jax.numpy as jnp
import numpy as np
import pytest

from massband.utils import unwrap_positions


@pytest.fixture()
def wrapped_trajectory():
    """Create a wrapped trajectory for testing."""
    # Create a simple trajectory with periodic boundary conditions
    cell = np.eye(3)  # 1 Angstrom cubic cell
    pos1 = [0.5, 0.5, 0.5]  # Start at the center of the cell
    pos2 = [0.9, 0.9, 0.9]  # Move close to the edge of the cell
    pos3 = [0.1, 0.1, 0.1]  # Assume movement across the boundary

    # particle moving twice across the boundary
    return (
        np.array([pos1, pos2, pos3, pos1, pos2, pos3, pos1, pos2, pos3]),
        np.array([cell] * 9),
        np.array([np.linalg.inv(cell)] * 9),
    )


def test_unwrap_positions(wrapped_trajectory):
    positions, cell, inv_cell = wrapped_trajectory
    # raise ValueError(f"Unwrap positions test: {positions.shape}, {cell.shape}, {inv_cell.shape}")

    # Unwrap the positions
    unwrapped_positions = unwrap_positions(positions, cell, inv_cell)

    # Check if the unwrapped positions are correct
    expected_unwrapped = np.array(
        [
            [0.5, 0.5, 0.5],
            [0.9, 0.9, 0.9],
            [1.1, 1.1, 1.1],
            [1.5, 1.5, 1.5],
            [1.9, 1.9, 1.9],
            [2.1, 2.1, 2.1],
            [2.5, 2.5, 2.5],
            [2.9, 2.9, 2.9],
            [3.1, 3.1, 3.1],
        ]
    )

    assert jnp.allclose(unwrapped_positions, expected_unwrapped)
