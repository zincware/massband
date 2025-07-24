# from massband import KinisiEinsteinHelfandIonicConductivity
import os

import pytest

import massband

ec_emc_results = {
    "ionic_conductivity": 8.488344970206363,
    "std": 2.208432207994494,
    "credible_interval_68": [6.324395513492948, 10.723095286633788],
    "credible_interval_95": [4.174445072166971, 12.820516702137388],
    "asymmetric_uncertainty": [
        2.163949456713415,
        2.2347503164274247,
    ],
}


def test_KinisiEinsteinHelfandIonicConductivity(tmp_path, ec_emc, ec_emc_smiles):
    os.chdir(tmp_path)

    diff = massband.KinisiEinsteinHelfandIonicConductivity(
        file=ec_emc,
        sampling_rate=1000,
        time_step=0.5,  # fs
        start_dt=5000,
        structures=ec_emc_smiles,
    )
    diff.run()

    # Check that all expected keys are present
    system_results = diff.results["system"]

    assert system_results["ionic_conductivity"] == pytest.approx(
        ec_emc_results["ionic_conductivity"], rel=0.1
    )
    assert system_results["std"] == pytest.approx(ec_emc_results["std"], rel=0.1)
    assert system_results["credible_interval_68"] == pytest.approx(
        ec_emc_results["credible_interval_68"], rel=0.1
    )
    assert system_results["credible_interval_95"] == pytest.approx(
        ec_emc_results["credible_interval_95"], rel=0.1
    )
    assert system_results["asymmetric_uncertainty"] == pytest.approx(
        ec_emc_results["asymmetric_uncertainty"], rel=0.1
    )
