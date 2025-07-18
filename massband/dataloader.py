import dataclasses
from pathlib import Path
import znh5md
import rdkit2ase
import typing as t
import jax.numpy as jnp
from jax import jit
import logging
import ase
from collections import defaultdict

log = logging.getLogger(__name__)
# TODO: determine ideal batch size by chunk size in the h5 file

@jit
def unwrap_positions(
    pos: jnp.ndarray, cell: jnp.ndarray, inv_cell: jnp.ndarray
) -> jnp.ndarray:
    frac = jnp.einsum("ij,faj->fai", inv_cell, pos)
    delta_frac = jnp.diff(frac, axis=0)
    delta_frac -= jnp.round(delta_frac)
    frac_unwrapped = jnp.concatenate(
        [frac[:1], frac[:1] + jnp.cumsum(delta_frac, axis=0)], axis=0
    )
    pos_unwrapped = jnp.einsum("ij,faj->fai", cell, frac_unwrapped)
    return pos_unwrapped

@jit
def wrap_positions(pos: jnp.ndarray, cells: jnp.ndarray, inv_cells: jnp.ndarray) -> jnp.ndarray:
    frac = jnp.einsum("ij,faj->fai", inv_cells, pos)
    frac_wrapped = frac % 1.0
    pos_wrapped = jnp.einsum("ij,faj->fai", cells, frac_wrapped)
    return pos_wrapped

def _get_indices(
    frame: ase.Atoms,
    structures: list[str] | None = None,
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
    ...     file="trajectory.h5",
    ...     batch_size=100,
    ...     structures=["CCO", "O"],  # ethanol and water
    ...     com=True,
    ...     wrap=True
    ... )
    >>> 
    >>> for batch_data, cell, inv_cell in loader:
    ...     # batch_data["CCO"] has shape (100, n_ethanol_molecules, 3)
    ...     # batch_data["O"] has shape (100, n_water_molecules, 3)
    ...     ethanol_positions = batch_data["CCO"]
    ...     water_positions = batch_data["O"]
    ...     # Process the batch...
    
    Process only atomic elements (no molecular grouping):
    
    >>> loader = TimeBatchedLoader(
    ...     file="trajectory.h5",
    ...     batch_size=50,
    ...     structures=None,  # Process by element
    ...     com=False,        # Don't calculate COM
    ...     wrap=False
    ... )
    >>> 
    >>> for batch_data, cell, inv_cell in loader:
    ...     # batch_data["C"] has shape (50, n_carbon_atoms, 3)
    ...     # batch_data["O"] has shape (50, n_oxygen_atoms, 3)
    ...     carbon_positions = batch_data["C"]
    ...     oxygen_positions = batch_data["O"]
    
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

    def __post_init__(self):
        if not self.fixed_cell:
            raise NotImplementedError("Non-fixed cell handling is not implemented yet.")
        
        self.handler = znh5md.IO(self.file, variable_shape=False, include=["position", "box"])
        
        # Determine the effective stop index and total number of frames to be processed.
        effective_stop = self.stop if self.stop is not None else len(self.handler)
        self.total_frames = len(range(self.start, effective_stop, self.step))

        if self.total_frames == 0:
            log.warning("The specified start, stop, and step result in zero frames to process.")
            return

        # If using memory mode, perform one single, efficient slice operation to load data.
        if self.memory:
            log.info(f"Loading {self.total_frames} frames into memory using slice [{self.start}:{self.stop}:{self.step}]...")
            self.handler = self.handler[self.start:self.stop:self.step]
            # After loading, treat the slice as a new trajectory with step=1 and start=0
            self.start = 0
            self.step = 1

        # Use the true first frame of the simulation for consistent properties
        self.first_frame_atoms = rdkit2ase.unwrap_structures(self.handler[0])
        self.first_frame_cell = jnp.array(self.first_frame_atoms.get_cell()[:])
        self.first_frame_inv_cell = jnp.linalg.inv(self.first_frame_cell)
        self.masses = jnp.array(self.first_frame_atoms.get_masses())
        self.indices = _get_indices(self.first_frame_atoms, self.structures)

        # Initialize unwrapping reference and iterator state
        self.iter_offset = 0
        self.last_processed_frame_positions = jnp.stack([self.first_frame_atoms.get_positions(), self.first_frame_atoms.get_positions()])
        
        log.info(f"Initialized loader for {self.total_frames} frames from {self.file}")

    def __len__(self):
        if not hasattr(self, 'total_frames'): return 0
        return (self.total_frames + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        self.iter_offset = 0
        return self
    
    def __next__(self) -> t.Tuple[dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray]:
        if not hasattr(self, 'total_frames') or self.iter_offset >= self.total_frames:
            raise StopIteration
        
        # --- BATCH SLICING LOGIC ---
        # Determine how many frames are in this batch
        frames_in_batch = min(self.batch_size, self.total_frames - self.iter_offset)
        
        # Calculate the start, stop, and step for this specific batch slice
        slice_start = self.start + self.iter_offset * self.step
        slice_stop = slice_start + frames_in_batch * self.step
        slice_step = self.step
        
        # Perform a single, efficient slice operation
        batch = self.handler[slice_start:slice_stop:slice_step]
        # --- END BATCH SLICING LOGIC ---

        if not batch:
            raise StopIteration

        batch_positions = jnp.array([x.get_positions() for x in batch])
        positions = jnp.concatenate(
            [self.last_processed_frame_positions, batch_positions], axis=0
        )
        unwrap_pos = unwrap_positions(
            positions, self.first_frame_cell, self.first_frame_inv_cell
        )
        
        unwrap_pos = unwrap_pos[2:]
        self.last_processed_frame_positions = jnp.stack(
            [self.first_frame_atoms.get_positions(), unwrap_pos[-1]]
        )

        # The rest of the processing logic remains the same...
        data = defaultdict(list)
        for structure, all_mols in self.indices.items():
            if not all_mols:
                continue
            mol_indices_array = jnp.array(all_mols)
            if not self.com:
                atom_indices = mol_indices_array.flatten()
                data[structure] = unwrap_pos[:, atom_indices, :]
            else:                
                mol_positions = unwrap_pos[:, mol_indices_array, :]
                masses = self.masses[mol_indices_array]
                total_mol_mass = jnp.sum(masses, axis=1)
                numerator = jnp.sum(mol_positions * masses[None, :, :, None], axis=2)
                com = numerator / total_mol_mass[None, :, None]
                data[structure] = com
        
        results = {}
        for structure, pos in data.items():
            if self.wrap:
                pos = wrap_positions(pos, self.first_frame_cell, self.first_frame_inv_cell)
            results[structure] = pos

        self.iter_offset += frames_in_batch

        return results, self.first_frame_cell, self.first_frame_inv_cell

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
    ...     file="trajectory.h5",
    ...     batch_size=100,  # Max 100 atoms per batch
    ...     structures=["CCO", "O"],  # ethanol and water
    ...     com=True,
    ...     wrap=True
    ... )
    >>> 
    >>> for batch_data, cell, inv_cell in loader:
    ...     for species, positions in batch_data.items():
    ...         # positions has shape (n_frames, n_molecules_in_batch, 3)
    ...         print(f"{species}: {positions.shape}")
    ...         # Process all temporal data for this species batch...
    
    Process by atomic elements with full trajectory for each batch:
    
    >>> loader = SpeciesBatchedLoader(
    ...     file="trajectory.h5",
    ...     batch_size=50,   # Max 50 atoms per batch
    ...     structures=None, # Process by element
    ...     com=False,       # Individual atom coordinates
    ...     wrap=False
    ... )
    >>> 
    >>> for batch_data, cell, inv_cell in loader:
    ...     for element, positions in batch_data.items():
    ...         # positions has shape (n_frames, n_atoms_in_batch, 3)
    ...         print(f"{element}: {positions.shape}")
    
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
    - Memory usage scales with trajectory length × atoms per batch, making it less
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

    def __post_init__(self):
        """
        Initializes the loader by identifying species and preparing batches of
        atom indices for lazy loading.
        """
        if not self.fixed_cell:
            raise NotImplementedError("Non-fixed cell handling is not implemented yet.")

        self.handler = znh5md.IO(self.file, variable_shape=False, include=["position", "box"])
        if self.memory:
            log.info(f"Loading {self.file} into memory ...")
            self.handler = self.handler[:] # TODO use start, stop, step here
        
        effective_stop = self.stop if self.stop is not None else len(self.handler)
        self.total_frames = len(range(self.start, effective_stop, self.step))

        if self.total_frames == 0:
            log.warning("The specified start, stop, and step result in zero frames to process.")
            self.species_batches = []
            return

        # Get simulation properties from the first frame
        self.first_frame_atoms = rdkit2ase.unwrap_structures(self.handler[self.start])
        self.cell = jnp.array(self.first_frame_atoms.get_cell()[:])
        self.inv_cell = jnp.linalg.inv(self.cell)
        self.masses = jnp.array(self.first_frame_atoms.get_masses())
        
        # --- Prepare Species Batches (without loading data) ---
        log.info(f"Grouping species into batches where atom count <= {self.batch_size}...")
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
                    current_batch = []
                    current_atom_count = 0
                
                current_batch.append(mol_indices)
                current_atom_count += atoms_per_mol
            
            if current_batch:
                self.species_batches.append((structure, jnp.array(current_batch)))
        
        log.info(f"Initialized loader with {len(self.species_batches)} total species batches.")

    def __len__(self):
        """Returns the total number of species-based batches."""
        if not hasattr(self, 'species_batches'): return 0
        return len(self.species_batches)

    def __iter__(self):
        """Resets the iterator over the species batches."""
        self.species_batch_iterator = iter(self.species_batches)
        return self

    def __next__(self) -> t.Tuple[dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray]:
        """
        Loads, unwraps, and processes the full trajectory for the next batch of species.
        """
        try:
            structure, mol_indices_array = next(self.species_batch_iterator)
        except StopIteration:
            raise StopIteration

        # --- Lazy Loading and Unwrapping ---
        # 1. Get the flat list of atom indices for this batch
        atom_indices_flat = mol_indices_array.flatten()
        
        # 2. Load only the required positions for these atoms over the specified time slice
        sliced_frames = self.handler[self.start:self.stop:self.step]
        raw_positions_list = [self.first_frame_atoms.get_positions()[atom_indices_flat]] + [frame.get_positions()[atom_indices_flat] for frame in sliced_frames[1:]]
        raw_positions = jnp.array(raw_positions_list)

        # 3. Unwrap the positions for this subset of atoms
        unwrapped_pos = unwrap_positions(raw_positions, self.cell, self.inv_cell)
        
        # --- Process Batch ---
        if not self.com:
            # If not calculating COM, the result is simply the unwrapped atom positions
            # Shape: (n_frames, n_atoms_in_batch, 3)
            pos = unwrapped_pos
        else:
            # Reshape positions to (frames, mols, atoms_per_mol, 3) for COM calculation
            n_mols_in_batch = mol_indices_array.shape[0]
            n_atoms_per_mol = mol_indices_array.shape[1]
            mol_positions = unwrapped_pos.reshape(self.total_frames, n_mols_in_batch, n_atoms_per_mol, 3)

            # Calculate Center of Mass trajectories
            masses = self.masses[mol_indices_array]
            total_mol_mass = jnp.sum(masses, axis=1)
            numerator = jnp.sum(mol_positions * masses[None, :, :, None], axis=2)
            pos = numerator / total_mol_mass[None, :, None]
        
        # Optionally wrap the final positions
        if self.wrap:
            pos = wrap_positions(pos, self.cell, self.inv_cell)
        
        results = {structure: pos}

        return results, self.cell, self.inv_cell
