import os

import pytest

import massband

ec_emc_results = {
    "F[P-](F)(F)(F)(F)F": {
        "diffusion_coefficient": 5.98157494352885e-07,
        "std": 9.639778343055821e-08,
        "credible_interval_68": [5.020045519080746e-07, 6.950658072384147e-07],
        "credible_interval_95": [4.069740011894767e-07, 7.858850110232055e-07],
        "asymmetric_uncertainty": [
            9.615294244481035e-08,
            9.69083128855297e-08,
        ],
        "occurrences": 16,
    },
    "[Li+]": {
        "diffusion_coefficient": 6.620708329653184e-07,
        "std": 9.351228601086382e-08,
        "credible_interval_68": [5.69321953987668e-07, 7.55613293658028e-07],
        "credible_interval_95": [4.782832320454286e-07, 8.479936196575174e-07],
        "asymmetric_uncertainty": [
            9.274887897765044e-08,
            9.354246069270959e-08,
        ],
        "occurrences": 16,
    },
    "C1COC(=O)O1": {
        "diffusion_coefficient": 2.071929333135169e-06,
        "std": 2.0807649385069142e-07,
        "credible_interval_68": [1.862128615089697e-06, 2.2764212311009853e-06],
        "credible_interval_95": [1.6570541880894185e-06, 2.4642949006289355e-06],
        "asymmetric_uncertainty": [
            2.0980071804547206e-07,
            2.0449189796581624e-07,
        ],
        "occurrences": 58,
    },
    "CCOC(=O)OC": {
        "diffusion_coefficient": 1.4579456010012004e-06,
        "std": 1.6806116108663198e-07,
        "credible_interval_68": [1.2903948057218276e-06, 1.6275124171461165e-06],
        "credible_interval_95": [1.1372759353371671e-06, 1.7861341503915456e-06],
        "asymmetric_uncertainty": [
            1.6755079527937287e-07,
            1.6956681614491605e-07,
        ],
        "occurrences": 44,
    },
    "C1=COC(=O)O1": {
        "diffusion_coefficient": 4.090805375538788e-06,
        "std": 1.4420795314508635e-06,
        "credible_interval_68": [2.6640559339602086e-06, 5.553053674803551e-06],
        "credible_interval_95": [1.4000519173775337e-06, 7.0726017543553795e-06],
        "asymmetric_uncertainty": [
            1.4267494415785793e-06,
            1.4622482992647632e-06,
        ],
        "occurrences": 3,
    },
    "COC(=O)OC": {
        "diffusion_coefficient": 1.7800537527054853e-06,
        "std": 1.8097746924210685e-07,
        "credible_interval_68": [1.5954004427582697e-06, 1.9609803686378305e-06],
        "credible_interval_95": [1.43033005596045e-06, 2.13132006633799e-06],
        "asymmetric_uncertainty": [
            1.846533099472156e-07,
            1.8092661593234518e-07,
        ],
        "occurrences": 54,
    },
}


def test_KinisiSelfDiffusion(tmp_path, ec_emc, ec_emc_smiles):
    os.chdir(tmp_path)

    diff = massband.KinisiSelfDiffusion(
        file=ec_emc,
        sampling_rate=1000,
        time_step=0.5,  # fs
        start_dt=5000,
        structures=ec_emc_smiles,
    )
    diff.run()

    for smile, expected in ec_emc_results.items():
        result = diff.results[smile]
        assert result["diffusion_coefficient"] == pytest.approx(
            expected["diffusion_coefficient"], rel=0.1
        )
        assert result["std"] == pytest.approx(expected["std"], rel=0.1)
        assert result["credible_interval_68"] == pytest.approx(
            expected["credible_interval_68"], rel=0.1
        )
        assert result["credible_interval_95"] == pytest.approx(
            expected["credible_interval_95"], rel=0.1
        )
        assert result["asymmetric_uncertainty"] == pytest.approx(
            expected["asymmetric_uncertainty"], rel=0.1
        )
        assert result["occurrences"] == expected["occurrences"]
