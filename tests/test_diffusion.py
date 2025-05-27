import pytest
import numpy as np
import pint
from ase import Atoms
import massband  # Assuming your class is here
import znh5md
from pathlib import Path

ureg = pint.UnitRegistry()


@pytest.fixture
def diffusive_md(tmp_path) -> Path:
    """Create a trajectory with a fixed diffusion coefficient.

    Returns:
        List[ase.Atoms]: A trajectory where particles follow Brownian motion
        with the specified diffusion coefficient.
    """
    # Parameters
    D = 1.0 * ureg.angstrom**2 / ureg.picosecond
    steps = 2_000
    dt = 0.5 * ureg.fs
    particles = 20

    rng = np.random.default_rng(42)  # For reproducibility

    # Convert quantities to atomic units
    D_val = D.to("angstrom**2/fs").magnitude
    dt_val = dt.to("fs").magnitude

    # Initialize system
    frames = []
    pos = np.zeros((particles, 3))  # Start all particles at origin

    max_disp_per_dim = np.sqrt(
        2 * D_val * dt_val * steps
    )  # Total displacement in one dimension for 1000 steps
    # A generous cell size
    cell_size = max_disp_per_dim * 5  # Factor 5 for safety, adjust as needed
    cell = np.diag([cell_size, cell_size, cell_size])  # Cubic cell

    # Generate Brownian motion trajectory
    for _ in range(steps):
        # Random displacement with variance 2DΔt in each dimension
        displacement = rng.normal(
            loc=0.0,  # Mean of 0
            scale=np.sqrt(2 * D_val * dt_val),
            size=(particles, 3),
        )
        pos += displacement

        # Create ASE Atoms object for this step
        atoms = Atoms(
            f"H{particles}", positions=pos, cell=cell, pbc=True
        )  # Add cell and pbc
        atoms.info["time"] = _ * dt_val
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
    )

    diff.run()
    # Now we expect results to be calculated, not None
    assert diff.results is not None
    assert diff.results[1]["diffusion_coefficient"] == pytest.approx(1.0, rel=0.1)
