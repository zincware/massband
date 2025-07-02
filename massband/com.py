import logging
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import ase
import jax.numpy as jnp
import rdkit2ase
import znh5md

from massband.utils import unwrap_positions, wrap_positions

log = logging.getLogger(__name__)


def load_unwrapped_frames(
    file: Path | str,
) -> Tuple[list[ase.Atoms], jnp.ndarray, jnp.ndarray]:
    """Load ASE frames from H5MD and unwrap their positions."""
    io = znh5md.IO(file, variable_shape=False, include=["position", "box"])
    frames: list[ase.Atoms] = io[:]
    log.info(f"Loaded {len(frames)} frames from {file}")
    positions = jnp.stack([atoms.positions for atoms in frames])
    cells = jnp.stack([atoms.cell[:] for atoms in frames])
    inv_cells = jnp.linalg.inv(cells)
    positions = unwrap_positions(positions, cells, inv_cells)
    return frames, positions, cells


def identify_substructures(
    frame: ase.Atoms, structures: list[str] | None = None
) -> dict[str, list[tuple[int, ...]]]:
    """Return atom index groups by substructure SMILES or element symbol."""
    substructures = defaultdict(list)
    if structures:
        for structure in structures:
            indices = rdkit2ase.match_substructure(
                frame, smiles=structure, suggestions=structures
            )
            if indices:
                substructures[structure].extend(indices)
                log.info(f"Found {len(indices)} matches for {structure}")
    else:
        symbols = frame.get_chemical_symbols()
        for element in set(symbols):
            indices = jnp.array([i for i, sym in enumerate(symbols) if sym == element])
            if indices.size > 0:
                substructures[element].extend((i,) for i in indices)
                log.info(f"Found {len(indices)} matches for element {element}")
    return substructures


def compute_com_trajectories(
    positions: jnp.ndarray,
    masses: jnp.ndarray,
    substructures: dict[str, list[tuple[int, ...]]],
    cells: jnp.ndarray,
    wrap: bool = False,
) -> dict[str, jnp.ndarray]:
    """Compute center of mass trajectories per substructure."""
    com_positions = defaultdict(list)
    for structure, all_indices in substructures.items():
        for mol_indices in all_indices:
            mol_masses = jnp.array([masses[i] for i in mol_indices])
            mol_positions = positions[:, mol_indices]
            mass_sum = jnp.sum(mol_masses)
            weighted_positions = mol_positions * mol_masses[None, :, None]
            com = jnp.sum(weighted_positions, axis=1) / mass_sum
            com_positions[structure].append(com)

    com_positions = {
        structure: jnp.stack(coms, axis=1) for structure, coms in com_positions.items()
    }
    if wrap:
        com_positions = {
            structure: wrap_positions(coms, cells)
            for structure, coms in com_positions.items()
        }
    return com_positions


def center_of_mass_trajectories(
    file: Path | str, structures: list[str] | None = None, wrap: bool = False
) -> Tuple[dict[str, jnp.ndarray], jnp.ndarray]:
    """Compute center of mass trajectories from an H5MD file."""
    frames, positions, cells = load_unwrapped_frames(file)
    masses = jnp.array(frames[0].get_masses())
    substructures = identify_substructures(frames[0], structures)
    com_positions = compute_com_trajectories(
        positions, masses, substructures, cells, wrap=wrap
    )
    log.info(f"Found COM positions: { {k: v.shape for k, v in com_positions.items()} }")
    return com_positions, cells
