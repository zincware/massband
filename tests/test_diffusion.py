import os
from pathlib import Path

import pytest

import massband


def test_KinisiSelfDiffusion(tmp_path, bmim_bf4_vectra):
    os.chdir(tmp_path)

    diff = massband.KinisiSelfDiffusion(
        file=bmim_bf4_vectra,
        sampling_rate=100,
        time_step=0.5,  # fs
        start_dt=100,
        structures=["CCCCN1C=C[N+](=C1)C", "[B-](F)(F)(F)F"],
    )
    diff.run()
    bmim_results = diff.results["CCCCN1C=C[N+](=C1)C"]
    bf4_results = diff.results["[B-](F)(F)(F)F"]

    assert bmim_results["diffusion_coefficient"] == pytest.approx(3.300e-06, rel=0.1)
    assert bmim_results["std"] == pytest.approx(3.205e-07, rel=0.1)
    assert bmim_results["credible_interval_68"] == pytest.approx(
        [2.9490e-06, 3.6275e-06], rel=0.1
    )
    assert bmim_results["credible_interval_95"] == pytest.approx(
        [2.59601e-06, 3.9878e-06], rel=0.1
    )
    assert bmim_results["asymmetric_uncertainty"] == pytest.approx(
        [3.206e-07, 3.177e-07], rel=0.1
    )

    assert bf4_results["diffusion_coefficient"] == pytest.approx(4.40842e-06, rel=0.1)
    assert bf4_results["std"] == pytest.approx(4.95086e-07, rel=0.1)
    assert bf4_results["credible_interval_68"] == pytest.approx(
        [3.9232e-06, 4.9071e-06], rel=0.1
    )
    assert bf4_results["credible_interval_95"] == pytest.approx(
        [3.414929e-06, 5.37291e-06], rel=0.1
    )
    assert bf4_results["asymmetric_uncertainty"] == pytest.approx(
        [4.85219e-07, 4.98712e-07], rel=0.1
    )

