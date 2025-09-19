import dataclasses
import logging
import typing as t
from collections import defaultdict
from pathlib import Path

import ase
import jax.numpy as jnp
import rdkit2ase
import znh5md
from jax import jit, lax
from rdkit import Chem

log = logging.getLogger(__name__)


class LoaderOutput(t.TypedDict, total=False):
    """
    Defines the structure of the dictionary yielded by the data loaders.

    Using `total=False` means keys are optional and will only be present if
    requested in the loader's `properties` attribute.
    """

    position: dict[str, jnp.ndarray]
    velocity: dict[str, jnp.ndarray]
    cell: jnp.ndarray
    inv_cell: jnp.ndarray
    masses: dict[str, jnp.ndarray]
    indices: dict[str, jnp.ndarray]


@jit
def unwrap_positions_image_flags_batched(
    pos: jnp.ndarray,
    cell: jnp.ndarray,
    inv_cell: jnp.ndarray,
    prev_image_flags: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Unwraps batched positions using iterative image flags (for TimeBatchedLoader)."""
    frac = jnp.einsum("ij,faj->fai", inv_cell, pos)

    def _update_step(flags_prev, frac_curr_prev_tuple):
        frac_curr, frac_prev = frac_curr_prev_tuple
        delta_frac = frac_curr - frac_prev
        flags_curr = flags_prev - jnp.round(delta_frac).astype(jnp.int32)
        return flags_curr, frac_curr + flags_curr

    xs = (frac[1:], frac[:-1])
    final_flags, frac_unwrapped = lax.scan(_update_step, prev_image_flags, xs)
    pos_unwrapped = jnp.einsum("ij,faj->fai", cell, frac_unwrapped)
    return pos_unwrapped, final_flags


@jit
def unwrap_trajectory_image_flags(
    pos: jnp.ndarray, cell: jnp.ndarray, inv_cell: jnp.ndarray
) -> jnp.ndarray:
    """Unwraps a full trajectory slice using image flags (for SpeciesBatchedLoader)."""
    frac = jnp.einsum("ij,faj->fai", inv_cell, pos)
    initial_flags = jnp.zeros_like(frac[0], dtype=jnp.int32)

    def _update_step(flags_prev, frac_curr_prev_tuple):
        frac_curr, frac_prev = frac_curr_prev_tuple
        delta_frac = frac_curr - frac_prev
        flags_curr = flags_prev - jnp.round(delta_frac).astype(jnp.int32)
        return flags_curr, frac_curr + flags_curr

    xs = (frac[1:], frac[:-1])
    _, frac_unwrapped_rest = lax.scan(_update_step, initial_flags, xs)
    # The first frame has no displacement, so its unwrapped state is its original state.
    frac_unwrapped = jnp.concatenate([frac[:1], frac_unwrapped_rest], axis=0)
    pos_unwrapped = jnp.einsum("ij,faj->fai", cell, frac_unwrapped)
    return pos_unwrapped


# --- UTILITY FUNCTIONS ---


@jit
def wrap_positions(
    pos: jnp.ndarray, cells: jnp.ndarray, inv_cells: jnp.ndarray
) -> jnp.ndarray:
    frac = jnp.einsum("ij,faj->fai", inv_cells, pos)
    frac_wrapped = frac % 1.0
    pos_wrapped = jnp.einsum("ij,faj->fai", cells, frac_wrapped)
    return pos_wrapped


def _get_indices(
    frame: ase.Atoms, structures: list[str] | None = None
) -> dict[str, list[tuple[int, ...]]]:
    indices = defaultdict(list)
    if not structures:
        symbols = frame.get_chemical_symbols()
        for element in sorted(set(symbols)):
            atom_indices = [(i,) for i, sym in enumerate(symbols) if sym == element]
            indices[element] = atom_indices
        return dict(indices)
    for structure in structures:
        mol_matches = rdkit2ase.match_substructure(
            frame, smiles=structure, suggestions=structures
        )
        if not mol_matches:
            continue
        indices[structure] = list(mol_matches)
    return dict(indices)


@dataclasses.dataclass
class IndependentBatchedLoader:
    file: Path | str
    wrap: bool
    batch_size: int = 64
    structures: list[str] | None = None
    com: bool = True
    memory: bool = False
    start: int = 0
    stop: int | None = None
    step: int = 1
    properties: list[
        t.Literal["position", "velocity", "cell", "inv-cell", "masses", "indices"]
    ] = dataclasses.field(default_factory=lambda: ["position", "cell", "inv-cell"])

    def __post_init__(self):
        self.handler = znh5md.IO(
            self.file, variable_shape=False, include=["position", "velocity", "box"]
        )
        if self.batch_size != 1:
            log.warning(
                "Batch size must be 1 for IndependentBatchedLoader. Setting to 1."
            )
            self.batch_size = 1  # can't be any larger for inhomogeneous shapes.
        # Calculate total_frames before modifying handler for memory mode
        original_length = len(self.handler)
        effective_stop = self.stop if self.stop is not None else original_length
        self.total_frames = len(range(self.start, effective_stop, self.step))
        if self.memory:
            log.info(f"Loading {self.total_frames} frames into memory...")
            self.handler = self.handler[self.start : self.stop : self.step]
            self.start, self.step = 0, 1
        if self.total_frames == 0:
            log.warning("The specified start, stop, and step result in zero frames.")
            return

        # Initialize indices from the first frame like other loaders
        first_frame_raw = self.handler[0]
        self.first_frame_atoms = rdkit2ase.unwrap_structures(first_frame_raw)
        self.indices = _get_indices(self.first_frame_atoms, self.structures)

    def __len__(self):
        if not hasattr(self, "total_frames"):
            return 0
        return self.total_frames

    def __iter__(self):
        self.iter_offset = 0
        return self

    def __next__(self) -> LoaderOutput:
        if not hasattr(self, "total_frames") or self.iter_offset >= self.total_frames:
            raise StopIteration

        frames_in_batch = min(self.batch_size, self.total_frames - self.iter_offset)
        slice_start = self.start + self.iter_offset * self.step
        slice_stop = slice_start + frames_in_batch * self.step
        slice_step = self.step

        batch = self.handler[slice_start:slice_stop:slice_step]
        if not batch:
            raise StopIteration

        pos_dict = {}
        vel_dict = {}
        masses_dict = {}
        indices_dict = {}
        cells = []

        for frame_idx, atoms in enumerate(batch):
            atoms = rdkit2ase.unwrap_structures(atoms)
            cells.append(atoms.get_cell()[:])

            # Get indices for this frame
            try:
                frame_indices = _get_indices(atoms, self.structures)
            except ValueError as e:
                log.warning(f"Failed to get indices for frame {frame_idx}: {e}")
                # Skip this frame if we can't process it
                continue

            for structure, mol_indices_list in frame_indices.items():
                if not mol_indices_list:
                    continue

                # Collect positions, velocities, and masses for each molecule
                for mol_indices in mol_indices_list:
                    if self.wrap:
                        atoms.wrap()
                    positions = atoms.get_positions()[list(mol_indices)]
                    masses = atoms.get_masses()[list(mol_indices)]

                    pos_dict.setdefault(structure, []).append(positions)
                    masses_dict.setdefault(structure, []).append(masses)

                    # Handle velocities if available
                    if (
                        hasattr(atoms, "get_velocities")
                        and atoms.get_velocities() is not None
                    ):
                        velocities = atoms.get_velocities()[list(mol_indices)]
                        vel_dict.setdefault(structure, []).append(velocities)
                    elif "velocity" in self.properties:
                        # If velocity is requested but not available, use zero velocities
                        zero_vel = jnp.zeros_like(positions)
                        vel_dict.setdefault(structure, []).append(zero_vel)

                # Store indices (convert tuples to arrays for consistency)
                if structure not in indices_dict:
                    indices_dict[structure] = jnp.array(mol_indices_list)
        # Convert lists to arrays, handling inhomogeneous shapes
        results = {}
        position_results = {}
        velocity_results = {}
        masses_results = {}

        for k, v in pos_dict.items():
            try:
                # Try to convert to regular array first
                position_results[k] = jnp.array(v)
            except ValueError:
                # Handle inhomogeneous shapes by keeping as list of arrays
                position_results[k] = [jnp.array(pos) for pos in v]

        for k, v in vel_dict.items():
            try:
                # Try to convert to regular array first
                velocity_results[k] = jnp.array(v)
            except ValueError:
                # Handle inhomogeneous shapes by keeping as list of arrays
                velocity_results[k] = [jnp.array(vel) for vel in v]

        for k, v in masses_dict.items():
            try:
                # Try to convert to regular array first
                masses_results[k] = jnp.array(v)
            except ValueError:
                # Handle inhomogeneous shapes by keeping as list of arrays
                masses_results[k] = [jnp.array(mass) for mass in v]

        results["position"] = position_results
        if vel_dict:
            results["velocity"] = velocity_results
        results["masses"] = masses_results

        # Add indices to results if requested
        if "indices" in self.properties:
            results["indices"] = indices_dict
        if "cell" in self.properties:
            results["cell"] = jnp.array(cells)

        self.iter_offset += frames_in_batch

        return results


@dataclasses.dataclass
class TimeBatchedLoader:
    """
    A data loader for efficiently processing molecular dynamics trajectories in time-based batches.

    This loader processes trajectory data frame-by-frame in batches, making it memory-efficient
    for large trajectories. It handles position unwrapping, center-of-mass calculations, and
    optional wrapping of coordinates back into the simulation box.

    Parameters
    ----------
    file : Path | str
        Path to the H5MD trajectory file to load.
    wrap : bool
        Whether to wrap coordinates back into the simulation box after processing.
    batch_size : int, optional
        Number of frames to process in each batch. Default is 64.
    structures : list[str] | None, optional
        List of SMILES strings defining molecular structures to extract. If None,
        processes all atoms grouped by element. Default is None.
    fixed_cell : bool, optional
        Whether to assume a fixed simulation cell. Default is True.
    com : bool, optional
        Whether to calculate center-of-mass for molecular structures. Default is True.
    memory : bool, optional
        Whether to load the entire trajectory into memory for faster access. Default is False.
    start : int, optional
        Starting frame index (0-based). Default is 0.
    stop : int | None, optional
        Ending frame index (exclusive). If None, processes to the end. Default is None.
    step : int, optional
        Frame step size. Default is 1.

    Yields
    ------
    tuple[dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray]
        A tuple containing:
        - results : dict[str, jnp.ndarray]
            Dictionary mapping structure names to position arrays.
            Each array has shape (n_frames_in_batch, n_atoms_or_molecules, 3).
        - cell : jnp.ndarray
            Simulation cell matrix with shape (3, 3).
        - inv_cell : jnp.ndarray
            Inverse cell matrix with shape (3, 3).

    Examples
    --------
    Load a trajectory and process in batches of 100 frames:

    >>> loader = TimeBatchedLoader(
    ...     file=ec_emc,
    ...     batch_size=100,
    ...     structures=["C1COC(=O)O1", "CCOC(=O)OC"],
    ...     com=False,
    ...     wrap=True,
    ... )
    >>> for output in loader:
    ...     batch_data = output["position"]
    ...     cell = output["cell"]
    ...     inv_cell = output["inv_cell"]
    ...     _ = batch_data["C1COC(=O)O1"]
    ...     _ = batch_data["CCOC(=O)OC"]
    ...     break  # Just process first batch for example

    Process only atomic elements (no molecular grouping):

    >>> loader = TimeBatchedLoader(
    ...     file=ec_emc,
    ...     batch_size=50,
    ...     structures=None,  # Process by element
    ...     com=False,  # Don't calculate COM
    ...     wrap=False,
    ... )
    >>> for output in loader:
    ...     batch_data = output["position"]
    ...     cell = output["cell"]
    ...     inv_cell = output["inv_cell"]
    ...     carbon_positions = batch_data["C"]
    ...     fluorine_positions = batch_data["F"]
    ...     break  # Just process first batch for example

    Notes
    -----
    - When `com=True` and `structures` is provided, positions represent center-of-mass
      coordinates of the specified molecular structures.
    - When `com=False` or `structures=None`, positions represent individual atomic coordinates.
    - The loader automatically handles periodic boundary conditions through unwrapping
      and optional re-wrapping of coordinates.
    - For consistent results across different batch sizes, especially with `memory=True`,
      the loader uses global trajectory context for unwrapping calculations.
    """

    file: Path | str
    wrap: bool
    batch_size: int = 64
    structures: list[str] | None = None
    fixed_cell: bool = True
    com: bool = True
    memory: bool = False
    start: int = 0
    stop: int | None = None
    step: int = 1
    properties: list[
        t.Literal["position", "velocity", "cell", "inv-cell", "masses", "indices"]
    ] = dataclasses.field(default_factory=lambda: ["position", "cell", "inv-cell"])
    map_to_dict: bool = True

    def __post_init__(self):
        if not self.fixed_cell:
            raise NotImplementedError("Non-fixed cell handling is not implemented yet.")
        if not self.map_to_dict and self.com:
            raise ValueError(
                "Mapping to dict with com=True is not supported. "
                "Set map_to_dict=False or com=False."
            )

        self.handler = znh5md.IO(
            self.file, variable_shape=False, include=["position", "velocity", "box"]
        )
        effective_stop = self.stop if self.stop is not None else len(self.handler)
        self.total_frames = len(range(self.start, effective_stop, self.step))

        if self.total_frames == 0:
            log.warning("The specified start, stop, and step result in zero frames.")
            return

        # Get the first frame before memory slicing to ensure consistency
        first_frame_raw = self.handler[self.start]
        if self.memory:
            log.info(f"Loading {self.total_frames} frames into memory...")
            self.handler = self.handler[self.start : self.stop : self.step]
            self.start, self.step = 0, 1
        self.first_frame_atoms = rdkit2ase.unwrap_structures(first_frame_raw)
        self.first_frame_pos = jnp.array(self.first_frame_atoms.get_positions())
        self.first_frame_cell = jnp.array(self.first_frame_atoms.get_cell()[:])
        self.first_frame_inv_cell = jnp.linalg.inv(self.first_frame_cell)

        self.masses = jnp.array(self.first_frame_atoms.get_masses())
        self.indices = _get_indices(self.first_frame_atoms, self.structures)

        # State for iteration: last wrapped position and the integer image flags
        self.last_wrapped_pos = self.first_frame_pos
        self.image_flag_state = jnp.zeros(
            (len(self.first_frame_atoms), 3), dtype=jnp.int32
        )

        self.iter_offset = 0
        log.info(f"Initialized loader for {self.total_frames} frames from {self.file}")

    @property
    def first_frame_chem(self) -> Chem.Mol:
        """Get the first frame as an RDKit molecule for substructure matching."""
        return rdkit2ase.ase2rdkit(self.first_frame_atoms, suggestions=self.structures)

    def __len__(self):
        if not hasattr(self, "total_frames"):
            return 0
        return (self.total_frames + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        self.iter_offset = 0
        # Reset state for a new iteration to ensure reproducibility
        if hasattr(self, "first_frame_pos"):  # Only if loader was properly initialized
            self.last_wrapped_pos = self.first_frame_pos
            self.image_flag_state = jnp.zeros(
                (len(self.first_frame_atoms), 3), dtype=jnp.int32
            )
        return self

    def __next__(self) -> LoaderOutput:
        if not hasattr(self, "total_frames") or self.iter_offset >= self.total_frames:
            raise StopIteration

        frames_in_batch = min(self.batch_size, self.total_frames - self.iter_offset)
        slice_start = self.start + self.iter_offset * self.step
        slice_stop = slice_start + frames_in_batch * self.step
        slice_step = self.step

        batch = self.handler[slice_start:slice_stop:slice_step]
        if not batch:
            raise StopIteration

        batch = list(batch)
        if self.iter_offset == 0:
            # Always use the consistent first frame atoms for the first frame
            batch[0] = self.first_frame_atoms

        batch_positions = jnp.array([x.get_positions() for x in batch])

        # Extract velocities if requested
        batch_velocities = None
        if "velocity" in self.properties:
            velocities_list = []
            for atoms in batch:
                if (
                    hasattr(atoms, "get_velocities")
                    and atoms.get_velocities() is not None
                ):
                    velocities_list.append(atoms.get_velocities())
                else:
                    # Use zero velocities if not available
                    velocities_list.append(jnp.zeros_like(atoms.get_positions()))
            batch_velocities = jnp.array(velocities_list)

        # Prepend the last wrapped position from the previous batch to provide context
        positions_with_context = jnp.concatenate(
            [self.last_wrapped_pos[None, :], batch_positions], axis=0
        )

        # Call the image flag unwrapping function with the context and current state
        unwrapped_pos, new_image_flag_state = unwrap_positions_image_flags_batched(
            positions_with_context,
            self.first_frame_cell,
            self.first_frame_inv_cell,
            self.image_flag_state,
        )

        # Update the state for the next iteration
        self.image_flag_state = new_image_flag_state
        self.last_wrapped_pos = batch_positions[-1]

        # Build output dictionary based on requested properties
        output = {}

        if "position" in self.properties:
            if self.map_to_dict:
                position_data = defaultdict(list)
                for structure, all_mols in self.indices.items():
                    if not all_mols:
                        continue
                    mol_indices_array = jnp.array(all_mols)
                    if not self.com:
                        atom_indices = mol_indices_array.flatten()
                        position_data[structure] = unwrapped_pos[:, atom_indices, :]
                    else:
                        mol_positions = unwrapped_pos[:, mol_indices_array, :]
                        masses = self.masses[mol_indices_array]
                        total_mol_mass = jnp.sum(masses, axis=1)
                        numerator = jnp.sum(
                            mol_positions * masses[None, :, :, None], axis=2
                        )
                        com = numerator / total_mol_mass[None, :, None]
                        position_data[structure] = com

                position_results = {}
                for structure, pos in position_data.items():
                    if self.wrap:
                        pos = wrap_positions(
                            pos, self.first_frame_cell, self.first_frame_inv_cell
                        )
                    position_results[structure] = pos
                output["position"] = position_results
            else:
                # If not mapping to dict, return a single array
                output["position"] = unwrapped_pos
        if "velocity" in self.properties and batch_velocities is not None:
            velocity_data = defaultdict(list)
            for structure, all_mols in self.indices.items():
                if not all_mols:
                    continue
                mol_indices_array = jnp.array(all_mols)
                if not self.com:
                    atom_indices = mol_indices_array.flatten()
                    velocity_data[structure] = batch_velocities[:, atom_indices, :]
                else:
                    # Compute COM velocity: sum(m_i * v_i) / sum(m_i)
                    mol_velocities = batch_velocities[:, mol_indices_array, :]
                    masses = self.masses[mol_indices_array]
                    total_mol_mass = jnp.sum(masses, axis=1)
                    numerator = jnp.sum(mol_velocities * masses[None, :, :, None], axis=2)
                    com_vel = numerator / total_mol_mass[None, :, None]
                    velocity_data[structure] = com_vel

            velocity_results = {}
            for structure, vel in velocity_data.items():
                velocity_results[structure] = vel
            output["velocity"] = velocity_results

        if "masses" in self.properties:
            masses_data = {}
            for structure, all_mols in self.indices.items():
                if not all_mols:
                    continue
                mol_indices_array = jnp.array(all_mols)
                if not self.com:
                    # For individual atoms, return masses for each atom
                    atom_indices = mol_indices_array.flatten()
                    masses_data[structure] = self.masses[atom_indices]
                else:
                    # For center of mass, return total mass for each molecule
                    masses = self.masses[mol_indices_array]
                    total_mol_mass = jnp.sum(masses, axis=1)
                    masses_data[structure] = total_mol_mass
            output["masses"] = masses_data

        if "cell" in self.properties:
            output["cell"] = self.first_frame_cell

        if "inv-cell" in self.properties:
            output["inv_cell"] = self.first_frame_inv_cell

        if "indices" in self.properties:
            # Convert indices format to match expected structure
            indices_data = {}
            for structure, mol_indices_list in self.indices.items():
                if mol_indices_list:
                    indices_data[structure] = jnp.array(mol_indices_list)
            output["indices"] = indices_data

        self.iter_offset += frames_in_batch
        return output


@dataclasses.dataclass
class SpeciesBatchedLoader:
    """
    A data loader that batches trajectory data by molecular species or substructures.

    This loader groups atoms/molecules by species and processes them in batches based on
    atom count limits. It loads the full trajectory for each species batch, making it
    efficient for analyses that need all temporal data for specific molecular species.

    Parameters
    ----------
    file : Path | str
        Path to the H5MD trajectory file to load.
    wrap : bool
        Whether to wrap coordinates back into the simulation box after processing.
    batch_size : int, optional
        Maximum number of atoms per batch. Molecules are grouped until this limit
        is reached. Default is 64.
    structures : list[str] | None, optional
        List of SMILES strings defining molecular structures to extract. If None,
        processes all atoms grouped by element. Default is None.
    fixed_cell : bool, optional
        Whether to assume a fixed simulation cell. Default is True.
    com : bool, optional
        Whether to calculate center-of-mass for molecular structures. Default is True.
    start : int, optional
        Starting frame index (0-based). Default is 0.
    stop : int | None, optional
        Ending frame index (exclusive). If None, processes to the end. Default is None.
    step : int, optional
        Frame step size. Default is 1.
    memory : bool, optional
        Whether to load the entire trajectory into memory for faster access. Default is False.

    Yields
    ------
    tuple[dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray]
        A tuple containing:
        - results : dict[str, jnp.ndarray]
            Dictionary mapping structure names to position arrays.
            Each array has shape (n_frames, n_molecules_in_batch, 3) when com=True,
            or (n_frames, n_atoms_in_batch, 3) when com=False.
        - cell : jnp.ndarray
            Simulation cell matrix with shape (3, 3).
        - inv_cell : jnp.ndarray
            Inverse cell matrix with shape (3, 3).

    Examples
    --------
    Process molecular species in batches with atom count limit:

    >>> loader = SpeciesBatchedLoader(
    ...     file=ec_emc,
    ...     batch_size=100,  # Max 100 atoms per batch
    ...     structures=["C1COC(=O)O1", "CCOC(=O)OC"],
    ...     com=False,
    ...     wrap=True,
    ... )
    >>> for output in loader:
    ...     batch_data = output["position"]
    ...     cell = output["cell"]
    ...     inv_cell = output["inv_cell"]
    ...     for species, positions in batch_data.items():
    ...         # positions has shape (n_frames, n_atoms_in_batch, 3)
    ...         _ = positions.shape  # Process all temporal data for this species batch
    ...         break
    ...     break  # Just process first batch for example

    Process by atomic elements with full trajectory for each batch:

    >>> loader = SpeciesBatchedLoader(
    ...     file=ec_emc,
    ...     batch_size=50,  # Max 50 atoms per batch
    ...     structures=None,  # Process by element
    ...     com=False,  # Individual atom coordinates
    ...     wrap=False,
    ... )
    >>> for output in loader:  # doctest: +ELLIPSIS
    ...     batch_data = output["position"]
    ...     cell = output["cell"]
    ...     inv_cell = output["inv_cell"]
    ...     for element, positions in batch_data.items():
    ...         # positions has shape (n_frames, n_atoms_in_batch, 3)
    ...         pass  # Process each element batch
    ...     break  # Just process first batch for example

    Notes
    -----
    - This loader processes the full trajectory for each species batch, making it
      suitable for analyses requiring complete temporal information for specific species.
    - The batch_size parameter limits the number of atoms (not molecules) per batch.
    - When com=True, the loader calculates center-of-mass for molecular structures
      and returns molecular coordinates.
    - When com=False or structures=None, returns individual atomic coordinates.
    - The loader automatically handles periodic boundary conditions through unwrapping
      and optional re-wrapping of coordinates.
    - Memory usage scales with trajectory length x atoms per batch, making it less
      memory-efficient than TimeBatchedLoader for very long trajectories.
    """

    file: Path | str
    wrap: bool
    batch_size: int = 64
    structures: list[str] | None = None
    fixed_cell: bool = True
    com: bool = True
    start: int = 0
    stop: int | None = None
    step: int = 1
    memory: bool = False
    properties: list[
        t.Literal["position", "velocity", "cell", "inv-cell", "masses", "indices"]
    ] = dataclasses.field(default_factory=lambda: ["position", "cell", "inv-cell"])

    def __post_init__(self):
        if not self.fixed_cell:
            raise NotImplementedError("Non-fixed cell handling is not implemented yet.")

        self.handler = znh5md.IO(
            self.file, variable_shape=False, include=["position", "velocity", "box"]
        )
        effective_stop = self.stop if self.stop is not None else len(self.handler)
        self.total_frames = len(range(self.start, effective_stop, self.step))

        if self.memory:
            log.info(f"Loading {self.file} into memory ...")
            self.handler = self.handler[self.start : self.stop : self.step]
            self.start, self.step = 0, 1

        if self.total_frames == 0:
            log.warning("The specified start, stop, and step result in zero frames.")
            self.species_batches = []
            return

        self.first_frame_atoms = rdkit2ase.unwrap_structures(self.handler[self.start])
        self.cell = jnp.array(self.first_frame_atoms.get_cell()[:])
        self.inv_cell = jnp.linalg.inv(self.cell)
        self.masses = jnp.array(self.first_frame_atoms.get_masses())

        log.info(
            f"Grouping species into batches where atom count <= {self.batch_size}..."
        )
        self.indices = _get_indices(self.first_frame_atoms, self.structures)
        self.species_batches = []
        for structure, all_mols in self.indices.items():
            if not all_mols:
                continue
            atoms_per_mol = len(all_mols[0])
            current_batch = []
            current_atom_count = 0
            for mol_indices in all_mols:
                if current_atom_count + atoms_per_mol > self.batch_size and current_batch:
                    self.species_batches.append((structure, jnp.array(current_batch)))
                    current_batch, current_atom_count = [], 0
                current_batch.append(mol_indices)
                current_atom_count += atoms_per_mol
            if current_batch:
                self.species_batches.append((structure, jnp.array(current_batch)))
        log.info(
            f"Initialized loader with {len(self.species_batches)} total species batches."
        )

    def __len__(self):
        if not hasattr(self, "species_batches"):
            return 0
        return len(self.species_batches)

    def __iter__(self):
        self.species_batch_iterator = iter(self.species_batches)
        return self

    def __next__(self) -> LoaderOutput:
        try:
            structure, mol_indices_array = next(self.species_batch_iterator)
        except StopIteration:
            raise StopIteration

        atom_indices_flat = mol_indices_array.flatten()

        sliced_frames = self.handler[self.start : self.stop : self.step]
        sliced_frames = list(sliced_frames)
        sliced_frames[0] = self.first_frame_atoms

        raw_positions = jnp.array(
            [frame.get_positions()[atom_indices_flat] for frame in sliced_frames]
        )

        # Extract velocities if requested
        raw_velocities = None
        if "velocity" in self.properties:
            velocities_list = []
            for frame in sliced_frames:
                if (
                    hasattr(frame, "get_velocities")
                    and frame.get_velocities() is not None
                ):
                    velocities_list.append(frame.get_velocities()[atom_indices_flat])
                else:
                    # Use zero velocities if not available
                    velocities_list.append(
                        jnp.zeros_like(frame.get_positions()[atom_indices_flat])
                    )
            raw_velocities = jnp.array(velocities_list)

        # Unwrap the full trajectory slice for this species using the image flag method
        unwrapped_pos = unwrap_trajectory_image_flags(
            raw_positions, self.cell, self.inv_cell
        )

        if not self.com:
            pos = unwrapped_pos
            vel = raw_velocities if raw_velocities is not None else None
        else:
            n_mols_in_batch = mol_indices_array.shape[0]
            n_atoms_per_mol = mol_indices_array.shape[1]
            mol_positions = unwrapped_pos.reshape(
                self.total_frames, n_mols_in_batch, n_atoms_per_mol, 3
            )
            masses = self.masses[mol_indices_array]
            total_mol_mass = jnp.sum(masses, axis=1)
            numerator = jnp.sum(mol_positions * masses[None, :, :, None], axis=2)
            pos = numerator / total_mol_mass[None, :, None]

            # Compute COM velocity if velocities are available
            if raw_velocities is not None:
                mol_velocities = raw_velocities.reshape(
                    self.total_frames, n_mols_in_batch, n_atoms_per_mol, 3
                )
                vel_numerator = jnp.sum(mol_velocities * masses[None, :, :, None], axis=2)
                vel = vel_numerator / total_mol_mass[None, :, None]
            else:
                vel = None

        if self.wrap:
            pos = wrap_positions(pos, self.cell, self.inv_cell)

        # Build output dictionary based on requested properties
        output = {}

        if "position" in self.properties:
            output["position"] = {structure: pos}

        if "velocity" in self.properties and vel is not None:
            output["velocity"] = {structure: vel}

        if "masses" in self.properties:
            if not self.com:
                # For individual atoms, return masses for each atom
                masses_data = self.masses[atom_indices_flat]
            else:
                # For center of mass, return total mass for each molecule
                masses = self.masses[mol_indices_array]
                total_mol_mass = jnp.sum(masses, axis=1)
                masses_data = total_mol_mass
            output["masses"] = {structure: masses_data}

        if "cell" in self.properties:
            output["cell"] = self.cell

        if "inv-cell" in self.properties:
            output["inv_cell"] = self.inv_cell

        if "indices" in self.properties:
            # For SpeciesBatchedLoader, return indices for current batch
            output["indices"] = {structure: mol_indices_array}

        return output
