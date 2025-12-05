import os
import subprocess

import zntrack

import massband


def test_load_io_file(tmp_path, ec_emc):
    os.chdir(tmp_path)

    project = zntrack.Project()

    subprocess.check_call(["git", "init"])
    subprocess.check_call(["dvc", "init"])

    with project:
        data = massband.LoadIOFile(path=ec_emc)
        rdf = massband.RadialDistributionFunction(
            data=data.frames,
            stop=100,
            bin_width=0.1,
        )
    project.repro()

    assert len(rdf.rdf.keys()) == 21
