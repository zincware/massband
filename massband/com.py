import logging
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import ase
import jax.numpy as jnp
import rdkit2ase
import znh5md
from jax import vmap

from massband.utils import unwrap_positions, wrap_positions

log = logging.getLogger(__name__)


def center_of_mass(
    file: Path | str, structures: list[str] | None = None, wrap: bool = False
) -> Tuple[dict[str, jnp.ndarray], jnp.ndarray]:
    # TODO: need to ensure that in the first frame, all molecules are fully unwrapped!!
    io = znh5md.IO(file, variable_shape=False, include=["position", "box"])
    frames: list[ase.Atoms] = io[:]

    log.info(f"Loaded {len(frames)} frames from {file}")
    positions = jnp.stack([atoms.positions for atoms in frames])
    cells = jnp.stack([atoms.cell[:] for atoms in frames])
    masses = jnp.array(frames[0].get_masses())
    inv_cells = jnp.linalg.inv(cells)
    log.info(f"Positions shape: {positions.shape}, Cells shape: {cells.shape}")
    positions = jnp.transpose(positions, (1, 0, 2))
    positions = vmap(lambda x: unwrap_positions(x, cells, inv_cells))(positions)
    positions = jnp.transpose(positions, (1, 0, 2))
    log.info(f"Unwrapped positions shape: {positions.shape}")

    # TODO: all of this could also go to utils? E.g. a get_center_of_mass positions function
    substructures = defaultdict(list)
    # a dict of list[tuple[int, ...]]
    if structures:
        log.info(f"Searching for substructures in {len(structures)} patterns")
        for structure in structures:
            indices = rdkit2ase.match_substructure(
                frames[0],
                smiles=structure,
                suggestions=structures,
            )
            if indices:
                substructures[structure].extend(indices)
                log.info(f"Found {len(indices)} matches for {structure}")

    # TODO: move to utils
    com_positions = defaultdict(list)

    for structure, all_indices in substructures.items():
        log.info(f"Computing COM positions for {structure}")

        for mol_indices in all_indices:
            mol_masses = jnp.array([masses[i] for i in mol_indices])  # (n_atoms_in_mol,)
            mol_positions = positions[:, mol_indices]  # (n_frames, n_atoms_in_mol, 3)

            # Compute COM for each frame: weighted sum over atoms
            # Numerator: sum_i(m_i * r_i), Denominator: sum_i(m_i)
            mass_sum = jnp.sum(mol_masses)
            weighted_positions = (
                mol_positions * mol_masses[None, :, None]
            )  # broadcast to (n_frames, n_atoms, 3)
            com = jnp.sum(weighted_positions, axis=1) / mass_sum  # (n_frames, 3)

            com_positions[structure].append(com)

    com_positions = {
        structure: jnp.transpose(jnp.stack(coms), (1, 0, 2))
        for structure, coms in com_positions.items()
    }
    if wrap:
        com_positions = {
            structure: wrap_positions(coms, cells)
            for structure, coms in com_positions.items()
        }
    log.info(f"Found COM positions: { {k: v.shape for k, v in com_positions.items()} }")

    return com_positions, cells
