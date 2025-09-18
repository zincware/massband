import itertools
import logging
import typing as t
from functools import partial
from pathlib import Path

import ase
import jax.lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pint
import rdkit2ase
import znh5md
import zntrack
from ase.data import atomic_numbers
from jax import jit, vmap
from tqdm.auto import tqdm

log = logging.getLogger(__name__)


class RDFData(t.TypedDict):
    """A data structure for storing RDF statistics."""

    bin_centers: list[float]
    g_r: list[float]
    unit: str
    number_density_a: float
    number_density_b: float


def generate_sorted_pairs(structure_names: list[str]) -> list[tuple[str, str]]:
    """
    Generate and sort unique element/compound name pairs based on chemical priority.

    The sorting creates a canonical order (e.g., H-O, not O-H; Element-Compound,
    not Compound-Element) to avoid duplicate calculations.

    The sorting logic follows these rules:
    1. Element pairs are sorted by atomic number.
    2. Element-compound pairs prioritize the element.
    3. Compound-compound pairs are sorted alphabetically.
    """

    def is_element(name):
        return name in atomic_numbers

    def sort_key(pair):
        a, b = pair
        a_is_elem, b_is_elem = is_element(a), is_element(b)

        # Canonicalize order within the pair before determining the sort key
        if (
            (a_is_elem and b_is_elem and atomic_numbers[a] > atomic_numbers[b])
            or (not a_is_elem and b_is_elem)
            or (not a_is_elem and not b_is_elem and a > b)
        ):
            a, b = b, a

        if a_is_elem and b_is_elem:
            return 0, atomic_numbers[a], atomic_numbers[b]
        elif a_is_elem:
            return 1, atomic_numbers[a], b
        # This case is implicitly handled by the swap above
        # elif b_is_elem:
        #     return (1, atomic_numbers[b], a)
        else:
            return 2, a, b

    # Generate all unique pairs and sort them canonically
    pairs = list(itertools.combinations_with_replacement(structure_names, 2))
    return sorted(pairs, key=sort_key)


def _finite_size_corrected_shell_volume(
    bin_edges: jnp.ndarray, L: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute corrected shell volumes for RDF normalization due to finite box size.

    This uses a simplified correction assuming a cubic box of side length L.
    For non-cubic boxes, L should be the minimum box dimension.
    """
    r_lower = bin_edges[:-1]
    r_upper = bin_edges[1:]
    r = 0.5 * (r_lower + r_upper)
    dr = r_upper - r_lower

    L_mean = jnp.mean(L)
    r_scaled = r / L_mean

    def spherical_shell(r_val):
        return 4 * jnp.pi * r_val**2

    shell_area = jnp.where(
        r_scaled <= 0.5,
        spherical_shell(r),
        4 * jnp.pi * r**2 * (1 - (3 / 2) * r_scaled + (1 / 2) * r_scaled**3),
    )
    # Set volume to zero for radii where the sphere intersects the box corners
    shell_area = jnp.where(r_scaled > jnp.sqrt(3) / 2, 0.0, shell_area)

    return shell_area * dr


@partial(jit, static_argnames=["exclude_self"])
def mic_distances(frame_a, frame_b, frame_cell, exclude_self: bool):
    """Compute minimum image convention distances between two sets of points."""
    inv_cell = jnp.linalg.inv(frame_cell)
    # Note: ASE cell vectors are rows, so pos @ inv(cell) gives fractional coords.
    frac_a = frame_a @ inv_cell
    frac_b = frame_b @ inv_cell

    delta_frac = frac_a[:, None, :] - frac_b[None, :, :]
    delta_frac -= jnp.round(delta_frac)
    delta_cart = delta_frac @ frame_cell
    dists = jnp.linalg.norm(delta_cart, axis=-1)

    if exclude_self:
        # Avoids JIT errors with dynamic sizes by checking if it's a square matrix
        if dists.shape[0] == dists.shape[1]:
            dists = dists.at[jnp.diag_indices_from(dists)].set(jnp.inf)

    return dists


@partial(jit, static_argnames=["exclude_self"])
def rdf_hist_single_frame(pos_a, pos_b, frame_cell, exclude_self, bin_edges):
    """Compute distance histogram for a single frame."""
    dists = mic_distances(pos_a, pos_b, frame_cell, exclude_self=exclude_self)
    return jnp.histogram(dists.flatten(), bins=bin_edges)[0]


def compute_rdf(
    positions_a: jnp.ndarray,
    positions_b: jnp.ndarray,
    cell: jnp.ndarray,
    bin_edges: jnp.ndarray,
    batch_size: int = 100,
    exclude_self: bool = False,
    progress_callback: t.Callable[[int], None] = None,
):
    """
    Compute the radial distribution function (RDF) g(r) for a trajectory.

    Uses numerically stable normalization to avoid overflow issues with large datasets.
    """
    n_frames = positions_a.shape[0]
    n_a = positions_a.shape[1]
    n_b = positions_b.shape[1]
    hist_sum = jnp.zeros(
        len(bin_edges) - 1, dtype=jnp.float32
    )  # Use float32 for accumulation

    # Process in batches to manage memory
    n_batches = (n_frames + batch_size - 1) // batch_size
    for i in range(n_batches):
        start_idx, end_idx = i * batch_size, min((i + 1) * batch_size, n_frames)

        batch_pos_a = positions_a[start_idx:end_idx]
        batch_pos_b = positions_b[start_idx:end_idx]
        batch_cell = cell[start_idx:end_idx]

        # Vectorize histogram calculation over the batch of frames
        batch_histograms = vmap(rdf_hist_single_frame, in_axes=(0, 0, 0, None, None))(
            batch_pos_a, batch_pos_b, batch_cell, exclude_self, bin_edges
        )
        hist_sum += jnp.sum(batch_histograms, axis=0).astype(jnp.float32)

        # Update progress if callback provided
        if progress_callback:
            progress_callback(1)

    # --- Numerically Stable Normalization ---
    # Instead of computing (n_pairs * n_frames) / (mean_volume * shell_volume)
    # We compute: hist / ((n_pairs / mean_volume) * n_frames * shell_volume)
    # This avoids large intermediate values

    volume = jnp.linalg.det(cell)
    mean_volume = jnp.mean(volume)

    # Use minimum box dimension for finite-size correction
    min_box_lengths = jnp.min(jnp.linalg.norm(cell, axis=2), axis=1)
    shell_volume = _finite_size_corrected_shell_volume(bin_edges, L=min_box_lengths)

    # Compute number densities
    number_density_a = n_a / mean_volume
    number_density_b = n_b / mean_volume

    # Compute pair density (pairs per unit volume) to avoid overflow
    if exclude_self:
        # For same-species RDF, we have N*(N-1) pairs per frame
        pair_density = (n_a * (n_a - 1)) / mean_volume
    else:
        # For different species, we have N_a * N_b pairs per frame
        pair_density = (n_a * n_b) / mean_volume

    # Now normalize: g(r) = hist / (pair_density * n_frames * shell_volume)
    # We can safely compute this step-by-step to avoid overflow

    # First normalize by number of frames (this reduces the magnitude)
    hist_per_frame = hist_sum / n_frames

    # Then normalize by pair density and shell volume
    # This is equivalent to the original formula but avoids large intermediate products
    norm_factor = pair_density * shell_volume

    # Avoid division by zero for empty bins or zero-volume shells
    g_r = jnp.where(norm_factor > 1e-9, hist_per_frame / norm_factor, 0)

    return g_r, number_density_a, number_density_b


@partial(jit, static_argnames=["n_molecules", "n_atoms_per_mol"])
def _calculate_pbc_aware_com_for_frame(
    frame_positions, frame_cell, atom_indices, masses, n_molecules, n_atoms_per_mol
):
    """
    JIT-compiled function to compute PBC-aware wrapped COMs for all molecules in one frame.
    Uses a pseudo-space method to handle molecules split across periodic boundaries.
    This version correctly handles triclinic (non-orthogonal) cells.
    """
    mol_positions = frame_positions[atom_indices]

    # Convert to fractional coordinates using the inverse cell matrix
    inv_cell = jnp.linalg.inv(frame_cell)
    s_coords = mol_positions @ inv_cell

    def compute_com_dimension(s_coords_dim, masses_norm):
        # Map fractional coordinates [0, 1] to angles [0, 2*pi]
        theta = s_coords_dim * (2 * jnp.pi)
        # Compute the center of mass in this pseudo-space
        xi = jnp.cos(theta)
        zeta = jnp.sin(theta)
        xi_bar = jnp.sum(masses_norm * xi, axis=1)
        zeta_bar = jnp.sum(masses_norm * zeta, axis=1)
        # Convert the average angle back to a fractional coordinate
        theta_bar = jnp.arctan2(-zeta_bar, -xi_bar) + jnp.pi
        return theta_bar / (2 * jnp.pi)

    total_mass_per_mol = jnp.sum(masses)
    masses_norm = masses[None, :] / total_mass_per_mol
    masses_norm = jnp.broadcast_to(masses_norm, (n_molecules, len(masses)))

    # Compute COM for each dimension in fractional coordinates
    com_scaled = jnp.stack(
        [compute_com_dimension(s_coords[:, :, i], masses_norm) for i in range(3)],
        axis=1,
    )

    # Convert fractional COM back to Cartesian coordinates
    return com_scaled @ frame_cell


class RadialDistributionFunction(zntrack.Node):
    """
    Calculate radial distribution functions (RDFs) for molecular dynamics trajectories.

    Computes g(r) for atom-atom or center-of-mass pairs with proper periodic boundary
    condition handling and finite-size corrections.

    Parameters
    ----------
    file : str | Path
        Path to trajectory file in H5MD format.
    data: znh5md.IO | list[ase.Atoms]| None, default None
        znh5md.IO object for trajectory data, as an alternative to 'file'.
    structures : list of str or None, default None
        SMILES strings for molecular structures. If provided, computes COM-COM RDFs.
        If None, computes atom-atom RDFs grouped by element.
    bin_width : float, default 0.05
        Width of distance bins in Angstrom.
    start : int, default 0
        Starting frame index.
    stop : int or None, default None
        Ending frame index. If None, processes all frames.
    step : int, default 1
        Frame sampling interval.
    batch_size : int, default 4
        Number of frames to process in each batch for memory management.

    Attributes
    ----------
    rdf : dict
        Dictionary containing RDF data for each pair with keys:
        - 'bin_centers': List of bin center positions
        - 'g_r': List of RDF values
        - 'unit': Distance unit string
        - 'number_density_a': Number density of species A
        - 'number_density_b': Number density of species B
    figures_path : Path
        Directory containing RDF plots.
    data_path : Path
        Directory containing RDF data files.
    """

    file: str | Path | None = zntrack.deps_path()
    data: znh5md.IO | list[ase.Atoms] | None = zntrack.deps(None)
    structures: list[str] | None = zntrack.params(None)
    bin_width: float = zntrack.params(0.05)
    start: int = zntrack.params(0)
    stop: int | None = zntrack.params(None)
    step: int = zntrack.params(1)
    batch_size: int = zntrack.params(4)

    rdf: dict[str, RDFData] = zntrack.outs()
    figures_path: Path = zntrack.outs_path(zntrack.nwd / "figures")
    data_path: Path = zntrack.outs_path(zntrack.nwd / "data")

    def run(self):
        """Executes the RDF calculation workflow."""
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.figures_path.mkdir(parents=True, exist_ok=True)
        self.rdf = {}

        # --- Data Loading and Unit Setup ---
        if self.data is not None and self.file is not None:
            raise ValueError("Provide either 'data' or 'file', not both.")
        elif self.file is not None:
            io = znh5md.IO(self.file, include=["position", "box"])
        elif self.data is not None:
            io = self.data
            if isinstance(io, znh5md.IO):
                io.include = ["position", "box"]
        else:
            raise ValueError("Either 'file' or 'data' must be provided.")
        frames = io[self.start : self.stop : self.step]
        ureg = pint.UnitRegistry()
        position_unit = ureg.angstrom

        # --- Molecule Identification ---
        graph = rdkit2ase.ase2networkx(frames[0], suggestions=self.structures)
        molecules: dict[str, tuple[tuple[int, ...]]] = {}
        masses: dict[str, np.ndarray] = {}

        if not self.structures:
            log.info("No structures provided, grouping by element for atom-atom RDFs.")
            molecules = {
                k: tuple((i,) for i in v) for k, v in frames[0].symbols.indices().items()
            }
        else:
            log.info("Structures provided. Grouping atoms for COM-COM RDFs.")
            for structure in self.structures:
                matches = rdkit2ase.match_substructure(
                    rdkit2ase.networkx2ase(graph), smiles=structure
                )
                if not matches:
                    log.warning(f"No matches found for structure {structure}")
                    continue
                molecules[structure] = matches
                masses[structure] = np.array(frames[0].get_masses()[list(matches[0])])

        # --- Pre-computation Step: Get all particle positions (Atoms or COMs) ---
        all_positions = jnp.array([frame.get_positions() for frame in frames])
        all_cells = jnp.array([frame.get_cell().__array__() for frame in frames])
        particle_positions = {}

        if self.structures:
            log.info("Preprocessing: Calculating COMs for all frames...")
            for name, mol_indices_tuples in tqdm(
                molecules.items(), desc="Computing COMs"
            ):
                n_molecules = len(mol_indices_tuples)
                if n_molecules == 0:
                    continue
                n_atoms_per_mol = len(mol_indices_tuples[0])
                atom_indices = jnp.array(mol_indices_tuples)
                mol_masses = jnp.array(masses[name])

                # Process frames sequentially with jax.lax.map
                def process_frame(frame_args):
                    frame_positions, frame_cell = frame_args
                    return _calculate_pbc_aware_com_for_frame(
                        frame_positions,
                        frame_cell,
                        atom_indices,
                        mol_masses,
                        n_molecules,
                        n_atoms_per_mol,
                    )

                particle_positions[name] = jax.lax.map(
                    process_frame, (all_positions, all_cells)
                )
        else:
            log.info("Preprocessing: Gathering atomic positions...")
            for name, atom_indices_tuples in molecules.items():
                indices = jnp.array([idx[0] for idx in atom_indices_tuples])
                particle_positions[name] = all_positions[:, indices, :]

        # --- RDF Calculation Loop with Enhanced Progress Tracking ---
        structure_names = sorted(molecules.keys())
        pairs = generate_sorted_pairs(structure_names)

        # Calculate total number of batches for all pairs
        n_frames = len(frames)
        total_batches = sum(
            (n_frames + self.batch_size - 1) // self.batch_size for pair in pairs
        )

        # Create a single progress bar for all RDF calculations
        with tqdm(total=total_batches, desc="Computing RDFs", unit="batch") as pbar:
            for struct_a, struct_b in pairs:
                if (
                    struct_a not in particle_positions
                    or struct_b not in particle_positions
                ):
                    continue

                pair_key = f"{struct_a}-{struct_b}"
                pbar.set_postfix_str(f"Processing {pair_key}")

                pos_a = particle_positions[struct_a]
                pos_b = particle_positions[struct_b]

                # Define bins based on the smallest box size to avoid artifacts
                min_box_dim = np.min(np.linalg.norm(all_cells, axis=2))
                r_max = min_box_dim / 2.0
                bin_edges = np.arange(0, r_max, self.bin_width)
                bin_centers = bin_edges[:-1] + self.bin_width / 2.0

                # --- RDF Calculation with progress callback ---
                def update_progress(n):
                    pbar.update(n)

                g_r, num_dens_a, num_dens_b = compute_rdf(
                    pos_a,
                    pos_b,
                    all_cells,
                    jnp.array(bin_edges),
                    batch_size=self.batch_size,
                    exclude_self=(struct_a == struct_b),
                    progress_callback=update_progress,
                )
                g_r_np = np.asarray(g_r)

                # --- Save Data and Metrics ---
                data_file = self.data_path / f"rdf_{pair_key}.txt"
                np.savetxt(
                    data_file,
                    np.vstack([bin_centers, g_r_np]).T,
                    header=f"r ({position_unit:~P}), g(r)",
                )

                self.rdf[pair_key] = {
                    "bin_centers": bin_centers.tolist(),
                    "g_r": g_r_np.tolist(),
                    "unit": f"{position_unit:~P}",
                    "number_density_a": float(num_dens_a),
                    "number_density_b": float(num_dens_b),
                }

                # --- Plotting ---
                plt.figure(figsize=(10, 6))
                plt.plot(bin_centers, g_r_np, label=pair_key)
                plt.xlabel(f"r  / {position_unit:~P}")
                plt.ylabel("g(r)")
                plt.title(f"Radial Distribution Function: {pair_key}")
                plt.axhline(1.0, color="grey", linestyle="--", label="Ideal Gas g(r)=1")
                plt.grid(True, linestyle="--", alpha=0.6)
                plt.legend()
                plt.tight_layout()
                plt.savefig(self.figures_path / f"rdf_{pair_key}.png", dpi=300)
                plt.close()
