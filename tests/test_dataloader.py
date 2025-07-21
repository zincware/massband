from collections import defaultdict
from itertools import combinations

import jax.numpy as jnp
import pytest

from massband.dataloader import SpeciesBatchedLoader, TimeBatchedLoader

# TODO: use a trajectory with very fast diffusion, e.g. via vectra
# TODO: the trajectory must be wrapped for the test to make sen
# TODO: test that the first frame is correctly unwrapped


@pytest.mark.parametrize("structures", [["CCCCN1C=C[N+](=C1)C", "[B-](F)(F)(F)F"], None])
@pytest.mark.parametrize("wrap", [True, False])
def test_TimeBatchedLoader_batch_size(wrap, structures, bmim_bf4_vectra):
    batch_sizes = [1, 2, 3, 4, 512]
    data = {}

    for batch_size in batch_sizes:
        tbdl = TimeBatchedLoader(
            file=bmim_bf4_vectra,
            structures=structures,
            wrap=wrap,
            memory=True,
            batch_size=batch_size,
        )
        results = defaultdict(list)
        for batch, _, _ in tbdl:
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


@pytest.mark.parametrize("structures", [["CCCCN1C=C[N+](=C1)C", "[B-](F)(F)(F)F"], None])
@pytest.mark.parametrize("wrap", [True, False])
def test_SpeciesBatchedLoader_batch_size(wrap, structures, bmim_bf4_vectra):
    batch_sizes = [1, 2, 3, 4, 512]
    data = {}

    for batch_size in batch_sizes:
        sbdl = SpeciesBatchedLoader(
            file=bmim_bf4_vectra,
            structures=structures,
            wrap=wrap,
            memory=True,
            batch_size=batch_size,
        )
        results = defaultdict(list)
        for batch, _, _ in sbdl:
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


@pytest.mark.parametrize("tbdl_batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("sbdl_batch_size", [1, 2, 4, 8])
def test_species_equals_time(tbdl_batch_size, sbdl_batch_size, bmim_bf4_vectra):
    tbdl = TimeBatchedLoader(
        file=bmim_bf4_vectra,
        structures=["CCCCN1C=C[N+](=C1)C", "[B-](F)(F)(F)F"],
        wrap=False,
        memory=True,
        batch_size=tbdl_batch_size,
    )
    sbdl = SpeciesBatchedLoader(
        file=bmim_bf4_vectra,
        structures=["CCCCN1C=C[N+](=C1)C", "[B-](F)(F)(F)F"],
        wrap=False,
        memory=True,
        batch_size=sbdl_batch_size,
    )
    assert jnp.allclose(tbdl.first_frame_atoms.positions, sbdl.first_frame_atoms.positions, atol=1e-6)

    tbdl_data = defaultdict(list)
    sbdl_data = defaultdict(list)

    for batch, _, _ in tbdl:
        for species, positions in batch.items():
            tbdl_data[species].append(positions)

    for batch, _, _ in sbdl:
        for species, positions in batch.items():
            sbdl_data[species].append(positions)

    assert tbdl_data.keys() == sbdl_data.keys()
    assert len(tbdl_data) == 2
    for key in tbdl_data.keys():
        concat_tbdl = jnp.concatenate(tbdl_data[key], axis=0)
        concat_sbdl = jnp.concatenate(sbdl_data[key], axis=1)

        assert concat_tbdl.shape == concat_sbdl.shape
        assert jnp.allclose(concat_tbdl, concat_sbdl, atol=1e-6)


@pytest.mark.parametrize("start", [0, 100])
@pytest.mark.parametrize("step", [1, 8])
@pytest.mark.parametrize("stop", [None])
def test_species_equals_time_start_step_stop(start, step, stop, bmim_bf4_vectra):
    tbdl = TimeBatchedLoader(
        file=bmim_bf4_vectra,
        structures=["CCCCN1C=C[N+](=C1)C", "[B-](F)(F)(F)F"],
        wrap=False,
        memory=True,
        batch_size=100,
        start=start,
        step=step,
        stop=stop,
    )
    sbdl = SpeciesBatchedLoader(
        file=bmim_bf4_vectra,
        structures=["CCCCN1C=C[N+](=C1)C", "[B-](F)(F)(F)F"],
        wrap=False,
        memory=True,
        batch_size=100,
        start=start,
        step=step,
        stop=stop,
    )
    assert jnp.allclose(tbdl.first_frame_atoms.positions, sbdl.first_frame_atoms.positions, atol=1e-6)

    tbdl_data = defaultdict(list)
    sbdl_data = defaultdict(list)

    for batch, _, _ in tbdl:
        for species, positions in batch.items():
            tbdl_data[species].append(positions)

    for batch, _, _ in sbdl:
        for species, positions in batch.items():
            sbdl_data[species].append(positions)

    assert tbdl_data.keys() == sbdl_data.keys()
    assert len(tbdl_data) == 2
    for key in tbdl_data.keys():
        concat_tbdl = jnp.concatenate(tbdl_data[key], axis=0)
        concat_sbdl = jnp.concatenate(sbdl_data[key], axis=1)

        assert concat_tbdl.shape == concat_sbdl.shape
        assert jnp.allclose(concat_tbdl, concat_sbdl, atol=1e-6)
