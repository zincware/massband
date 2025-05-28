import pytest
import numpy as np
import pint
from ase import Atoms
import massband
from massband.diffusion.utils import compute_msd_direct, compute_msd_fft
import numpy.testing as npt
import tidynamics
import znh5md
from pathlib import Path

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


@pytest.fixture
def diffusive_md(tmp_path, diffusive_positions) -> Path:
    """Create a trajectory with a fixed diffusion coefficient.

    Returns:
        List[ase.Atoms]: A trajectory where particles follow Brownian motion
        with the specified diffusion coefficient.
    """
    positions, dt_val = diffusive_positions
    particles = positions.shape[1]
    cell = np.eye(3) * 10.0
    dt_val = 0.5

    frames = []
    # Generate Brownian motion trajectory
    for position in positions:
        # Random displacement with variance 2DΔt in each dimension

        # Create ASE Atoms object for this step
        atoms = Atoms(
            f"H{particles}", positions=position, cell=cell, pbc=True
        )  # Add cell and pbc
        frames.append(atoms)

    io = znh5md.IO(
        filename=tmp_path / "diffusive_trajectory.h5",
        timestep=dt_val,
        variable_shape=False,
    )
    io.extend(frames)

    return tmp_path / "diffusive_trajectory.h5"


def test_EinsteinSelfDiffusion(diffusive_md):
    """Test the Einstein self-diffusion coefficient calculation."""
    diff = massband.EinsteinSelfDiffusion(
        file=diffusive_md,
        sampling_rate=1,
        timestep=0.5,  # fs
        batch_size=1,
        fit_window=(0.05, 0.95),  # we don't have a ballistic regime
    )

    diff.run()
    # Now we expect results to be calculated, not None
    assert diff.results is not None
    assert diff.results[1]["diffusion_coefficient"] == pytest.approx(1.0, rel=0.1)


def test_compute_msd_direct(diffusive_positions):
    """Test the direct computation of Mean Squared Displacement."""
    positions, _ = diffusive_positions
    msd_1 = compute_msd_direct(positions[:, 0])
    msd_2 = tidynamics.msd(positions[:, 0])
    msd_3 = compute_msd_fft(positions[:, 0])
    # Check that the two methods yield similar results, we ignore the first 10 and last 10 points to avoid edge effects of the fft
    npt.assert_allclose(msd_1[10:-10], msd_2[10:-10], rtol=5e-4)
    npt.assert_allclose(msd_1[10:-10], msd_3[10:-10], rtol=5e-4)
