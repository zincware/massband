import logging
from functools import partial
from pathlib import Path

import ase
import jax.lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import rdkit2ase
import vesin
import znh5md
import zntrack
from jax import jit
from scipy import signal
from scipy.optimize import curve_fit
from tqdm.auto import tqdm

log = logging.getLogger(__name__)


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


def compute_cluster_lifetime_correlation_fft(
    cluster_memberships: list[list[set]], max_lag: int = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the cluster lifetime autocorrelation function using FFT for better performance.

    This method tracks whether pairs of particles remain in the same cluster over time,
    providing insights into cluster stability and exchange dynamics in ionic liquids.

    Based on methods from:
    - Köhler et al., Faraday Discuss. 2018, 206, 339-351 (doi:10.1039/C7FD00166E)
    - Zhang & Maginn, J. Phys. Chem. Lett. 2015, 6, 700-705 (doi:10.1021/acs.jpclett.5b00003)

    Parameters
    ----------
    cluster_memberships : list of list of sets
        For each frame, list of sets containing particle indices in each cluster
    max_lag : int, optional
        Maximum lag time for correlation. If None, uses half the trajectory length

    Returns
    -------
    lags : np.ndarray
        Array of lag times
    correlation : np.ndarray
        Cluster lifetime autocorrelation function normalized to C(0)=1
    """
    n_frames = len(cluster_memberships)
    if max_lag is None:
        max_lag = n_frames // 2

    # Get all unique particles across all frames
    all_particles = set()
    for frame_clusters in cluster_memberships:
        for cluster in frame_clusters:
            all_particles.update(cluster)

    if len(all_particles) == 0:
        return np.arange(max_lag), np.zeros(max_lag)

    particles = sorted(all_particles)
    n_particles = len(particles)
    particle_to_idx = {p: i for i, p in enumerate(particles)}

    # Create binary time series for each particle pair
    # 1 if in same cluster, 0 otherwise
    pair_series = []

    # Process pairs in batches for memory efficiency
    for i in tqdm(range(n_particles), desc="Building pair correlations", leave=False):
        for j in range(i + 1, n_particles):
            series = np.zeros(n_frames)

            for t, frame_clusters in enumerate(cluster_memberships):
                # Check if particles i and j are in same cluster
                for cluster in frame_clusters:
                    if particles[i] in cluster and particles[j] in cluster:
                        series[t] = 1
                        break

            if np.any(series > 0):  # Only keep pairs that cluster at least once
                pair_series.append(series)

    if len(pair_series) == 0:
        return np.arange(max_lag), np.zeros(max_lag)

    # Compute autocorrelation using FFT for each pair
    correlations = []
    for series in tqdm(pair_series, desc="Computing autocorrelations", leave=False):
        # Use scipy's correlate with FFT method
        acf = signal.correlate(series, series, mode="full", method="fft")
        # Take only positive lags and normalize
        acf = acf[n_frames - 1 : n_frames - 1 + max_lag]

        # Normalize by the number of overlapping points
        norm = np.arange(n_frames, n_frames - max_lag, -1)
        acf = acf / norm

        # Normalize by initial value if non-zero
        if acf[0] > 0:
            acf = acf / acf[0]
            correlations.append(acf)

    if len(correlations) == 0:
        return np.arange(max_lag), np.zeros(max_lag)

    # Average over all pairs
    mean_correlation = np.mean(correlations, axis=0)

    return np.arange(max_lag), mean_correlation


def compute_cluster_statistics(clusters_over_time: list[list[set]]) -> dict:
    """
    Compute various cluster statistics from trajectory.

    Parameters
    ----------
    clusters_over_time : list of list of sets
        For each frame, list of sets containing particle indices in each cluster

    Returns
    -------
    stats : dict
        Dictionary containing various cluster statistics including size distributions,
        mean/max sizes, and fraction of particles in different cluster categories
    """
    stats = {}

    # Cluster size distribution
    all_sizes = []
    mean_sizes = []
    max_sizes = []
    n_clusters = []

    for frame_clusters in tqdm(
        clusters_over_time, desc="Computing statistics", leave=False
    ):
        sizes = [len(c) for c in frame_clusters]
        if sizes:
            all_sizes.extend(sizes)
            mean_sizes.append(np.mean(sizes))
            max_sizes.append(np.max(sizes))
            n_clusters.append(len(sizes))
        else:
            mean_sizes.append(0)
            max_sizes.append(0)
            n_clusters.append(0)

    stats["all_cluster_sizes"] = all_sizes
    stats["mean_cluster_size"] = mean_sizes
    stats["max_cluster_size"] = max_sizes
    stats["n_clusters"] = n_clusters

    # Fraction of particles in clusters of different sizes
    # Categories based on typical ionic liquid clustering patterns
    size_fractions = {1: [], 2: [], "3-5": [], "6+": []}

    for frame_clusters in clusters_over_time:
        total_particles = sum(len(c) for c in frame_clusters)
        if total_particles > 0:
            size_fractions[1].append(
                sum(len(c) for c in frame_clusters if len(c) == 1) / total_particles
            )
            size_fractions[2].append(
                sum(len(c) for c in frame_clusters if len(c) == 2) / total_particles
            )
            size_fractions["3-5"].append(
                sum(len(c) for c in frame_clusters if 3 <= len(c) <= 5) / total_particles
            )
            size_fractions["6+"].append(
                sum(len(c) for c in frame_clusters if len(c) >= 6) / total_particles
            )
        else:
            for key in size_fractions:
                size_fractions[key].append(0)

    stats["size_fractions"] = size_fractions

    return stats


def exponential_decay(t, A, tau, C):
    """Exponential decay function for fitting cluster lifetime correlations."""
    return A * np.exp(-t / tau) + C


class ClusterAnalysis(zntrack.Node):
    """
    Analyze cluster formation and dynamics in molecular dynamics trajectories.

    Specialized for ionic liquid systems with comprehensive cluster statistics including
    lifetimes, size distributions, and temporal correlations. This analysis is particularly
    relevant for understanding ion association, transport properties, and structural
    heterogeneity in ionic liquids.

    Parameters
    ----------
    file : str or Path | None
        Path to trajectory file in H5MD format.
    data: znh5md.IO | list[ase.Atoms] | None, default None
        znh5md.IO object for trajectory data, as an alternative to 'file'.
    structures : list of str or None, default None
        SMILES strings for molecular structures. If provided, computes COM-based clustering.
        If None, computes atom-based clustering grouped by element.
    cutoff_factor : float, default 1.5
        Factor to multiply minimum distance for cluster cutoff determination.
        Typically 1.2-1.5 for contact ion pairs in ionic liquids.
    start : int, default 0
        Starting frame index.
    stop : int or None, default None
        Ending frame index. If None, processes all frames.
    step : int, default 1
        Frame sampling interval.
    batch_size : int, default 4
        Number of frames to process in each batch for memory management.
    window_size : int, default 10
        Window size for running average of cluster properties.

    Attributes
    ----------
    cluster_stats : dict
        Dictionary containing comprehensive cluster statistics including:
        - Size distributions and temporal evolution
        - Cluster lifetime autocorrelation functions
        - Percolation indicators
        - Ion association fractions
    figures_path : Path
        Directory containing analysis plots.
    data_path : Path
        Directory containing analysis data files.
    """

    file: str | Path | None = zntrack.deps_path()
    data: znh5md.IO | list[ase.Atoms] | None = zntrack.deps(None)
    structures: list[str] | None = zntrack.params(None)
    cutoff_factor: float = zntrack.params(1.5)
    start: int = zntrack.params(0)
    stop: int | None = zntrack.params(None)
    step: int = zntrack.params(1)
    batch_size: int = zntrack.params(4)
    window_size: int = zntrack.params(10)

    # cluster_stats: dict = zntrack.outs()
    figures_path: Path = zntrack.outs_path(zntrack.nwd / "figures")
    data_path: Path = zntrack.outs_path(zntrack.nwd / "data")

    def run(self):
        """Executes the cluster analysis workflow."""
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.figures_path.mkdir(parents=True, exist_ok=True)
        self.cluster_stats = {}

        # --- Data Loading ---
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
        log.info(f"Loaded {len(frames)} frames for analysis")

        # --- Molecule Identification ---
        graph = rdkit2ase.ase2networkx(frames[0], suggestions=self.structures)
        molecules: dict[str, tuple[tuple[int, ...]]] = {}
        masses: dict[str, np.ndarray] = {}

        if not self.structures:
            log.info(
                "No structures provided, grouping by element for atom-based clustering."
            )
            molecules = {
                k: tuple((i,) for i in v) for k, v in frames[0].symbols.indices().items()
            }
        else:
            log.info("Structures provided. Grouping atoms for COM-based clustering.")
            for structure in self.structures:
                matches = rdkit2ase.match_substructure(
                    rdkit2ase.networkx2ase(graph), smiles=structure
                )
                if not matches:
                    log.warning(f"No matches found for structure {structure}")
                    continue
                molecules[structure] = matches
                masses[structure] = np.array(frames[0].get_masses()[list(matches[0])])

        # --- Pre-computation: Get all particle positions (Atoms or COMs) ---
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

        # --- Cluster Analysis ---
        nl = vesin.NeighborList(cutoff=10.0, full_list=True)

        for mol in sorted(molecules.keys()):
            trajectory_positions = particle_positions[mol]
            log.info(f"Analyzing {mol}: {trajectory_positions.shape}")

            minimum_distance: float | None = None
            clusters_over_time = []

            # Collect clusters for each frame
            for idx, (position, cell) in enumerate(
                tqdm(
                    zip(trajectory_positions, all_cells),
                    total=len(trajectory_positions),
                    desc=f"Computing clusters for {mol}",
                )
            ):
                i, j, d = nl.compute(position, cell, periodic=True, quantities="ijd")

                if minimum_distance is None and len(d) > 0:
                    minimum_distance = float(jnp.min(d)) * self.cutoff_factor
                    log.info(f"Cluster cutoff distance set to {minimum_distance:.3f} Å")

                if minimum_distance is not None:
                    d_ij = np.full((len(position), len(position)), np.inf)
                    d_ij[i, j] = d
                    np.fill_diagonal(d_ij, 0.0)

                    connectivity_matrix = np.zeros(
                        (len(position), len(position)), dtype=int
                    )
                    connectivity_matrix[d_ij <= minimum_distance] = 1

                    G = nx.from_numpy_array(connectivity_matrix)
                    frame_clusters = list(nx.connected_components(G))
                    clusters_over_time.append(frame_clusters)
                else:
                    clusters_over_time.append([])

            # Compute cluster statistics
            log.info(f"Computing cluster statistics for {mol}...")
            stats = compute_cluster_statistics(clusters_over_time)

            # Compute cluster lifetime autocorrelation using FFT
            log.info(f"Computing cluster lifetime autocorrelation for {mol}...")
            lags, correlation = compute_cluster_lifetime_correlation_fft(
                clusters_over_time
            )
            stats["lifetime_correlation_lags"] = lags
            stats["lifetime_correlation"] = correlation

            # Try to fit exponential decay to extract characteristic lifetime
            try:
                valid = ~np.isnan(correlation) & (correlation > 0.01)
                if np.sum(valid) > 3:
                    popt, _ = curve_fit(
                        exponential_decay,
                        lags[valid],
                        correlation[valid],
                        p0=[1.0, len(lags) / 10, 0.0],
                        maxfev=5000,
                    )
                    stats["cluster_lifetime_tau"] = (
                        popt[1] * self.step
                    )  # Convert to frame units
                    log.info(
                        f"Characteristic cluster lifetime: {stats['cluster_lifetime_tau']:.1f} frames"
                    )
            except:
                stats["cluster_lifetime_tau"] = None

            self.cluster_stats[mol] = stats

            # --- Plotting ---
            log.info(f"Generating plots for {mol}...")

            # 1. Cluster size distribution
            fig, ax = plt.subplots(figsize=(8, 6))
            if stats["all_cluster_sizes"]:
                max_size = max(stats["all_cluster_sizes"])
                ax.hist(
                    stats["all_cluster_sizes"],
                    bins=np.arange(1, max_size + 2) - 0.5,
                    density=True,
                    alpha=0.7,
                    edgecolor="black",
                )
                ax.set_xlabel("Cluster Size", fontsize=12)
                ax.set_ylabel("Probability", fontsize=12)
                ax.set_title(f"Cluster Size Distribution for {mol}", fontsize=14)
                ax.grid(True, alpha=0.3)
            fig_path = (
                self.figures_path
                / f"cluster_size_distribution_{mol.replace('/', '_')}.png"
            )
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            # 2. Number of clusters over time with running average
            fig, ax = plt.subplots(figsize=(10, 6))
            frames_idx = np.arange(len(stats["n_clusters"])) * self.step
            ax.plot(
                frames_idx, stats["n_clusters"], alpha=0.3, color="gray", label="Raw data"
            )

            # Running average using scipy
            if len(stats["n_clusters"]) > self.window_size:
                running_avg = signal.convolve(
                    stats["n_clusters"],
                    np.ones(self.window_size) / self.window_size,
                    mode="valid",
                )
                avg_frames = frames_idx[: len(running_avg)]
                ax.plot(
                    avg_frames,
                    running_avg,
                    color="blue",
                    linewidth=2,
                    label=f"Running average (window={self.window_size})",
                )

            ax.set_xlabel("Frame", fontsize=12)
            ax.set_ylabel("Number of Clusters", fontsize=12)
            ax.set_title(f"Number of Clusters Over Time for {mol}", fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig_path = (
                self.figures_path / f"n_clusters_over_time_{mol.replace('/', '_')}.png"
            )
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            # 3. Mean cluster size over time with running average
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(
                frames_idx,
                stats["mean_cluster_size"],
                alpha=0.3,
                color="gray",
                label="Raw data",
            )

            if len(stats["mean_cluster_size"]) > self.window_size:
                running_avg = signal.convolve(
                    stats["mean_cluster_size"],
                    np.ones(self.window_size) / self.window_size,
                    mode="valid",
                )
                avg_frames = frames_idx[: len(running_avg)]
                ax.plot(
                    avg_frames,
                    running_avg,
                    color="red",
                    linewidth=2,
                    label=f"Running average (window={self.window_size})",
                )

            ax.set_xlabel("Frame", fontsize=12)
            ax.set_ylabel("Mean Cluster Size", fontsize=12)
            ax.set_title(f"Mean Cluster Size Over Time for {mol}", fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig_path = (
                self.figures_path
                / f"mean_cluster_size_over_time_{mol.replace('/', '_')}.png"
            )
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            # 4. Cluster lifetime autocorrelation function
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(lags * self.step, correlation, "o-", markersize=3)
            if stats["cluster_lifetime_tau"] is not None:
                # Plot fitted exponential
                fit_y = exponential_decay(
                    lags, 1.0, stats["cluster_lifetime_tau"] / self.step, 0
                )
                ax.plot(
                    lags * self.step,
                    fit_y,
                    "--",
                    label=f"τ = {stats['cluster_lifetime_tau']:.1f} frames",
                )
            ax.set_xlabel("Lag Time (frames)", fontsize=12)
            ax.set_ylabel("Cluster Lifetime Autocorrelation", fontsize=12)
            ax.set_title(f"Cluster Lifetime Correlation for {mol}", fontsize=14)
            ax.set_ylim(-0.1, 1.1)
            ax.grid(True, alpha=0.3)
            if stats["cluster_lifetime_tau"] is not None:
                ax.legend()
            fig_path = (
                self.figures_path
                / f"cluster_lifetime_correlation_{mol.replace('/', '_')}.png"
            )
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            # 5. Fraction of particles in clusters of different sizes
            fig, ax = plt.subplots(figsize=(10, 6))
            for size_key, fractions in stats["size_fractions"].items():
                if len(fractions) > self.window_size:
                    running_avg = signal.convolve(
                        fractions,
                        np.ones(self.window_size) / self.window_size,
                        mode="valid",
                    )
                    avg_frames = frames_idx[: len(running_avg)]
                    ax.plot(
                        avg_frames, running_avg, linewidth=2, label=f"Size {size_key}"
                    )

            ax.set_xlabel("Frame", fontsize=12)
            ax.set_ylabel("Fraction of Particles", fontsize=12)
            ax.set_title(f"Particle Distribution by Cluster Size for {mol}", fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            fig_path = (
                self.figures_path / f"cluster_size_fractions_{mol.replace('/', '_')}.png"
            )
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            # 6. Percolation analysis - check for system-spanning clusters
            percolation_indicator = []
            for frame_clusters in clusters_over_time:
                max_size = max([len(c) for c in frame_clusters]) if frame_clusters else 0
                total_particles = len(trajectory_positions[0])
                percolation_indicator.append(max_size / total_particles)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(frames_idx, percolation_indicator, alpha=0.5)
            ax.axhline(
                y=0.5, color="r", linestyle="--", label="Percolation threshold (50%)"
            )
            ax.set_xlabel("Frame", fontsize=12)
            ax.set_ylabel("Largest Cluster Fraction", fontsize=12)
            ax.set_title(f"Percolation Indicator for {mol}", fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            fig_path = (
                self.figures_path / f"percolation_indicator_{mol.replace('/', '_')}.png"
            )
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            # Save numerical data
            np.savez(
                self.data_path / f"cluster_analysis_{mol.replace('/', '_')}.npz",
                n_clusters=stats["n_clusters"],
                mean_cluster_size=stats["mean_cluster_size"],
                max_cluster_size=stats["max_cluster_size"],
                all_cluster_sizes=stats["all_cluster_sizes"],
                lifetime_correlation_lags=stats["lifetime_correlation_lags"],
                lifetime_correlation=stats["lifetime_correlation"],
                cluster_lifetime_tau=stats["cluster_lifetime_tau"]
                if stats["cluster_lifetime_tau"]
                else np.nan,
                percolation_indicator=percolation_indicator,
            )

            log.info(f"Completed analysis for {mol}")
