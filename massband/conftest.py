"""pytest setup for `pytest --doctest-modules ipsuite/`."""

import os
import pathlib
import subprocess

import ase.io
import pytest
import rdkit2ase
import zntrack

import massband

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"


@pytest.fixture
def project(tmp_path: pathlib.Path):
    """
    A pytest fixture that creates a temporary directory,
    initializes git and dvc, and yields an ips.Project instance.
    """
    # Store the original directory to return to it later
    original_cwd = pathlib.Path.cwd()
    os.chdir(tmp_path)

    ethanol = rdkit2ase.smiles2conformers("CCO", numConfs=100)
    ase.io.write("ethanol.xyz", ethanol)

    # Setup: Initialize git and DVC
    try:
        subprocess.run(["git", "init"], check=True, capture_output=True)
        # Using --quiet to keep the output clean
        subprocess.run(["dvc", "init", "--quiet"], check=True, capture_output=True)

        # Yield the project instance for the test to use
        yield zntrack.Project()

    finally:
        # Teardown: Go back to the original directory
        os.chdir(original_cwd)


@pytest.fixture
def ec_emc_smiles() -> list[str]:
    smiles = {
        "PF6": "F[P-](F)(F)(F)(F)F",
        "Li": "[Li+]",
        "EC": "C1COC(=O)O1",
        "EMC": "CCOC(=O)OC",
        "VC": "C1=COC(=O)O1",
        "DMC": "COC(=O)OC",
    }
    return list(smiles.values())


@pytest.fixture
def doctest_namespace(project):
    """
    A pytest fixture that adds `project` and `massband` and
    all .h5 files in the `data` directory to the doctest namespace.
    """
    namespace = {"project": project, "massband": massband}
    for file in DATA_DIR.glob("*.h5"):
        namespace[file.stem] = file
    return namespace
