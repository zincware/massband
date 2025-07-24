import numpy as np
import numpy.testing as npt
import pint
import pytest
import tidynamics

from massband.utils import compute_msd_direct, compute_msd_fft

ureg = pint.UnitRegistry()


@pytest.fixture
def diffusive_positions() -> np.ndarray:
    # Parameters
    D = 1.0 * ureg.angstrom**2 / ureg.picosecond
    steps = 100
    dt = 0.5 * ureg.fs
    particles = 20

    rng = np.random.default_rng(42)  # For reproducibility

    # Convert quantities to atomic units
    D_val = D.to("angstrom**2/fs").magnitude
    dt_val = dt.to("fs").magnitude

    # Initialize system
    frames = []
    pos = np.zeros((particles, 3))  # Start all particles at origin

    for _ in range(steps):
        # Random displacement with variance 2DΔt in each dimension
        displacement = rng.normal(
            loc=0.0,  # Mean of 0
            scale=np.sqrt(2 * D_val * dt_val),
            size=(particles, 3),
        )
        pos += displacement
        frames.append(pos.copy())

    return np.array(frames, dtype=np.float32), dt_val


def test_compute_msd_direct(diffusive_positions):
    """Test the direct computation of Mean Squared Displacement."""
    positions, _ = diffusive_positions
    msd_1 = compute_msd_direct(positions[:, 0])
    msd_2 = tidynamics.msd(positions[:, 0])
    msd_3 = compute_msd_fft(positions[:, 0])
    # Check that the two methods yield similar results, we ignore the first 10 and last 10 points to avoid edge effects of the fft
    npt.assert_allclose(msd_1[10:-10], msd_2[10:-10], rtol=5e-4)
    npt.assert_allclose(msd_1[10:-10], msd_3[10:-10], rtol=5e-4)
