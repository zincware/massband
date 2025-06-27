import os
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pint
import pytest
import tidynamics

import massband
from massband.diffusion.utils import compute_msd_direct, compute_msd_fft

ureg = pint.UnitRegistry()

BMIM_BF4_FILE = (Path(__file__).parent.parent / "data" / "bmim_bf4.h5").resolve()


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


def test_KinisiSelfDiffusion(tmp_path):
    os.chdir(tmp_path)

    diff = massband.KinisiSelfDiffusion(
        file=BMIM_BF4_FILE,
        sampling_rate=100,
        time_step=0.5,  # fs
        start_dt=100,
        structures=["CCCCN1C=C[N+](=C1)C", "[B-](F)(F)(F)F"],
    )
    diff.run()
    bmim_results = diff.results["CCCCN1C=C[N+](=C1)C"]
    bf4_results = diff.results["[B-](F)(F)(F)F"]

    assert bmim_results["diffusion_coefficient"] == pytest.approx(1.423e-06, rel=0.1)
    assert bmim_results["std"] == pytest.approx(3.205e-07, rel=0.1)
    assert bmim_results["credible_interval_68"] == pytest.approx(
        [1.115e-06, 1.753e-06], rel=0.1
    )
    assert bmim_results["credible_interval_95"] == pytest.approx(
        [7.966e-07, 2.002e-06], rel=0.1
    )
    assert bmim_results["asymmetric_uncertainty"] == pytest.approx(
        [3.206e-07, 3.177e-07], rel=0.1
    )

    assert bf4_results["diffusion_coefficient"] == pytest.approx(9.018e-07, rel=0.1)
    assert bf4_results["std"] == pytest.approx(2.793e-07, rel=0.1)
    assert bf4_results["credible_interval_68"] == pytest.approx(
        [6.339e-07, 1.125e-06], rel=0.1
    )
    assert bf4_results["credible_interval_95"] == pytest.approx(
        [3.803e-07, 1.377e-06], rel=0.1
    )
    assert bf4_results["asymmetric_uncertainty"] == pytest.approx(
        [2.729e-07, 2.809e-07], rel=0.1
    )
