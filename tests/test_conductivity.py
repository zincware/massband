# from massband import KinisiEinsteinHelfandIonicConductivity
import massband
import os

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

    raise ValueError(diff.results)
