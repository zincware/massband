from collections import defaultdict
from itertools import combinations

import jax.numpy as jnp
import pytest

from massband.dataloader import SpeciesBatchedLoader, TimeBatchedLoader

# TODO: test that the first frame is correctly unwrapped


@pytest.mark.parametrize("structures", [True, False])
@pytest.mark.parametrize("wrap", [True, False])
def test_TimeBatchedLoader_batch_size(wrap, structures, ec_emc, ec_emc_smiles):
    batch_sizes = [1, 2, 3, 4, 512]
    data = {}

    for batch_size in batch_sizes:
        tbdl = TimeBatchedLoader(
            file=ec_emc,
            structures=ec_emc_smiles if structures else None,
            wrap=wrap,
            memory=True,
            stop=512,
            batch_size=batch_size,
        )
        results = defaultdict(list)
        for batch_output in tbdl:
            batch = batch_output["position"]
            for species, positions in batch.items():
                results[species].append(positions)

        data[batch_size] = {
            species: jnp.concatenate(positions, axis=0)
            for species, positions in results.items()
        }

    reference = data[batch_sizes[0]]
    tolerance = 1e-6
    for species in reference.keys():
        matches = []
        mismatches = []
        for a, b in combinations(batch_sizes, 2):
            equal = jnp.allclose(data[a][species], data[b][species], atol=tolerance)
            (matches if equal else mismatches).append((a, b))

        if mismatches:
            match_str = ", ".join(f"{x}=={y}" for x, y in matches) or "none"
            mismatch_str = ", ".join(f"{x}!={y}" for x, y in mismatches)
            raise AssertionError(
                f"Species '{species}' mismatch across batch sizes.\n"
                f"Equal: {match_str}\n"
                f"Not equal: {mismatch_str}"
            )


@pytest.mark.parametrize("structures", [True, False])
@pytest.mark.parametrize("wrap", [True, False])
def test_SpeciesBatchedLoader_batch_size(wrap, structures, ec_emc, ec_emc_smiles):
    batch_sizes = [1, 2, 3, 4, 512]
    data = {}

    for batch_size in batch_sizes:
        sbdl = SpeciesBatchedLoader(
            file=ec_emc,
            structures=ec_emc_smiles if structures else None,
            wrap=wrap,
            memory=True,
            stop=512,
            batch_size=batch_size,
        )
        results = defaultdict(list)
        for batch_output in sbdl:
            batch = batch_output["position"]
            for species, positions in batch.items():
                results[species].append(positions)

        data[batch_size] = {
            species: jnp.concatenate(positions, axis=1)
            for species, positions in results.items()
        }

    reference = data[batch_sizes[0]]
    tolerance = 1e-6
    for species in reference.keys():
        matches = []
        mismatches = []
        for a, b in combinations(batch_sizes, 2):
            equal = jnp.allclose(data[a][species], data[b][species], atol=tolerance)
            (matches if equal else mismatches).append((a, b))

        if mismatches:
            match_str = ", ".join(f"{x}=={y}" for x, y in matches) or "none"
            mismatch_str = ", ".join(f"{x}!={y}" for x, y in mismatches)
            raise AssertionError(
                f"Species '{species}' mismatch across batch sizes.\n"
                f"Equal: {match_str}\n"
                f"Not equal: {mismatch_str}"
            )


@pytest.mark.parametrize("wrap", [True, False])
def test_start_stop_step_consistency(ec_emc, wrap):
    """Test that the start/stop/step functionality is consistent."""
    tbl = TimeBatchedLoader(
        file=ec_emc,
        wrap=wrap,
        batch_size=64,
        start=10,
        step=5,
        stop=53,
        memory=False,
    )

    tbl_memory = TimeBatchedLoader(
        file=ec_emc,
        wrap=wrap,
        batch_size=64,
        start=10,
        step=5,
        stop=53,
        memory=True,
    )

    results = list(tbl)
    tbl_results = results[0]
    tbl_memory_results = list(tbl_memory)[0]
    assert len(results) == 1  # one batch
    assert set(tbl_results["position"]) == {"C", "F", "H", "Li", "O", "P"}

    sbl = SpeciesBatchedLoader(
        file=ec_emc,
        wrap=wrap,
        batch_size=2048,
        start=10,
        step=5,
        stop=53,
        memory=False,
    )

    sbl_memory = SpeciesBatchedLoader(
        file=ec_emc,
        wrap=wrap,
        batch_size=2048,
        start=10,
        step=5,
        stop=53,
        memory=True,
    )

    results = list(sbl)
    positions = {}
    for x in results:
        positions.update(x["position"])
    assert set(positions) == {"C", "F", "H", "Li", "O", "P"}

    positions_memory = {}
    for x in list(sbl_memory):
        positions_memory.update(x["position"])

    assert positions["C"].shape == (9, 521, 3)  # 9 frames, 521 atoms, 3 coordinates
    assert positions["F"].shape == (9, 96, 3)
    assert positions["P"].shape == (9, 16, 3)

    for key in tbl_results["position"]:
        assert jnp.allclose(
            tbl_results["position"][key],
            positions[key],
        )

        assert jnp.allclose(
            tbl_results["position"][key], tbl_memory_results["position"][key]
        )

        assert jnp.allclose(
            tbl_memory_results["position"][key],
            positions_memory[key],
        )


@pytest.mark.parametrize("wrap", [True, False])
def test_start_stop_step_consistency_structures(ec_emc, ec_emc_smiles, wrap):
    """Test that the start/stop/step functionality is consistent."""
    tbl = TimeBatchedLoader(
        file=ec_emc,
        wrap=wrap,
        batch_size=64,
        start=10,
        step=5,
        stop=53,
        memory=False,
        structures=ec_emc_smiles,
    )
    tbl_memory = TimeBatchedLoader(
        file=ec_emc,
        wrap=wrap,
        batch_size=64,
        start=10,
        step=5,
        stop=53,
        memory=True,
        structures=ec_emc_smiles,
    )

    results = list(tbl)
    tbl_results = results[0]
    tbl_memory_results = list(tbl_memory)[0]
    assert len(results) == 1  # one batch
    assert set(tbl_results["position"]) == set(ec_emc_smiles)

    sbl = SpeciesBatchedLoader(
        file=ec_emc,
        wrap=wrap,
        batch_size=2048,
        start=10,
        step=5,
        stop=53,
        memory=False,
        structures=ec_emc_smiles,
    )
    sbl_memory = SpeciesBatchedLoader(
        file=ec_emc,
        wrap=wrap,
        batch_size=2048,
        start=10,
        step=5,
        stop=53,
        memory=True,
        structures=ec_emc_smiles,
    )

    results = list(sbl)
    positions = {}
    positions_memory = {}
    for x in results:
        positions.update(x["position"])
    for x in list(sbl_memory):
        positions_memory.update(x["position"])
    assert set(positions) == set(ec_emc_smiles)

    assert positions["COC(=O)OC"].shape == (
        9,
        54,
        3,
    )  # 9 frames, 54 molecules, 3 coordinates
    assert positions["F[P-](F)(F)(F)(F)F"].shape == (9, 16, 3)

    for key in tbl_results["position"]:
        assert jnp.allclose(
            tbl_results["position"][key],
            positions[key],
        )
        assert jnp.allclose(
            tbl_results["position"][key], tbl_memory_results["position"][key]
        )
        assert jnp.allclose(
            tbl_memory_results["position"][key],
            positions_memory[key],
        )
