import os
from pathlib import Path

import pytest

import massband

BMIM_BF4_FILE = (Path(__file__).parent.parent / "data" / "bmim_bf4.h5").resolve()


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
