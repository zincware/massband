import contextlib
import logging
import os
import typing as t
from pathlib import Path

import ase
import h5py
import matplotlib.pyplot as plt
import numpy as np
import rdkit2ase
import scipp as sc
import scipy.stats as st
import znh5md
import zntrack
from kinisi.analyze import DiffusionAnalyzer

from massband.abc import ComparisonResults
from massband.comparison_utils import create_bar_comparison, create_overlay_plot
from massband.diffusion.types import DiffusionData
from massband.utils import sanitize_structure_name

log = logging.getLogger(__name__)


def make_hdf5_file_opener(
    self, path: str | Path | os.PathLike
) -> t.Callable[[], t.ContextManager[h5py.File]]:
    """Create a context manager to open an HDF5 file using the node file system."""

    @contextlib.contextmanager
    def _opener() -> t.Generator[h5py.File, None, None]:
        with self.state.fs.open(path, "rb") as f:
            yield h5py.File(f, "r")

    return _opener


class KinisiSelfDiffusion(zntrack.Node):
    """Compute self-diffusion coefficients using the kinisi library.

    Analyzes molecular dynamics trajectories to calculate self-diffusion coefficients
    and mean squared displacements for specified molecular structures.

    Parameters
    ----------
    file : Union[str, Path] | None
        Path to the trajectory file in h5md format.
    data: znh5md.IO | list[ase.Atoms] | None, default None
        znh5md.IO object for trajectory data, as an alternative to 'file'.
    structures : list[str] | None
        List of SMILES strings representing molecular structures to analyze.
        If None, analyzes all atomic species individually.
    start : int, default=0
        Starting frame index for trajectory analysis.
    stop : int | None, default=None
        Ending frame index. If None, uses all frames.
    step : int, default=1
        Frame step size for trajectory subsampling.
    time_step : float
        Simulation time step in femtoseconds.
    sampling_rate : int
        Number of simulation steps between saved trajectory frames.
    dt : tuple[float, float, float] | None, default=None
        Time interval parameters (start, stop, step) in femtoseconds.
        If None, uses kinisi defaults.
    start_dt : float
        Minimum time interval for diffusion coefficient fitting in femtoseconds.

    Attributes
    ----------
    data_path : Path
        Output directory for data files.
    figures_path : Path
        Output directory for plots.
    diffusion : dict[str, dict]
        Dictionary containing diffusion coefficient statistics and occurrences
        for each analyzed structure. Each entry contains:
        - mean: mean diffusion coefficient
        - std: standard deviation
        - var: variance
        - occurrences: number of molecules/ions of this type

    Examples
    --------
    >>> with project:
    ...     diff = massband.KinisiSelfDiffusion(
    ...         file=ec_emc,
    ...         time_step=0.5,
    ...         sampling_rate=1000,
    ...         structures=["[Li+]"],
    ...         start_dt=500_000,
    ...         step=1000,
    ...     )
    >>> project.repro()
    >>> diff.diffusion["[Li+]"].keys()
    dict_keys(['mean', 'std', 'var'])

    References
    ----------
    .. [1] https://kinisi.readthedocs.io/en/stable/
    """

    data: znh5md.IO | list[ase.Atoms] = zntrack.deps()
    structures: list[str] | None = zntrack.params()
    start: int = zntrack.params(0)
    stop: int | None = zntrack.params(None)
    step: int = zntrack.params(1)

    time_step: float = zntrack.params()  # in fs
    sampling_rate: int = zntrack.params()  # in number of frames

    data_path: Path = zntrack.outs_path(zntrack.nwd / "data")
    figures_path: Path = zntrack.outs_path(zntrack.nwd / "figures")
    dt: tuple[float, float, float] | None = zntrack.params(None)
    start_dt: float = zntrack.params()  # in fs

    diffusion: dict[str, DiffusionData] = zntrack.metrics()

    @staticmethod
    def compare(  # noqa: C901
        *nodes: "KinisiSelfDiffusion",
        labels: list[str] | None = None,
        structures: list[str] | None = None,
        use_plotly: bool = False,
    ) -> ComparisonResults:
        """Compare diffusion coefficients from multiple calculations.

        Parameters
        ----------
        *nodes : KinisiSelfDiffusion
            Multiple diffusion node instances to compare
        labels : list[str] | None
            Labels for each node (e.g., ["ML potential", "DFT", "Experiment"]).
            If None, uses node indices as labels.
        structures : list[str] | None
            Specific structures to compare (e.g., ["[Li+]", "EC"]).
            If None, compares all common structures across all nodes.
        use_plotly : bool
            If True, creates plotly figures; otherwise matplotlib.

        Returns
        -------
        ComparisonResults
            Dictionary containing comparative diffusion plots with keys:
            - "diffusion_coefficients": bar chart comparing D values
            - "relative_differences": bar chart showing relative % differences
            - "msd_{structure}": MSD overlay plot for each structure
        """
        if len(nodes) < 2:
            raise ValueError("At least two nodes are required for comparison")

        # Generate default labels if not provided
        if labels is None:
            labels = [
                node.name if node.name else f"Node {i + 1}"
                for i, node in enumerate(nodes)
            ]
        elif len(labels) != len(nodes):
            raise ValueError(
                f"Number of labels ({len(labels)}) must match number of nodes ({len(nodes)})"
            )

        # Find common structures across all nodes
        common_structures = set(nodes[0].diffusion.keys())
        for node in nodes[1:]:
            common_structures &= set(node.diffusion.keys())

        if not common_structures:
            raise ValueError("No common structures found across all nodes")

        # Filter to specific structures if requested
        if structures is not None:
            common_structures = set(structures) & common_structures
            if not common_structures:
                raise ValueError(
                    f"None of the requested structures {structures} are common to all nodes"
                )

        common_structures = sorted(common_structures)
        figures: dict[str, plt.Figure] = {}

        # Extract diffusion data
        diffusion_values: dict[str, list[float]] = {}
        diffusion_errors: dict[str, list[float]] = {}
        relative_differences: dict[str, list[float]] = {}

        for structure in common_structures:
            diffusion_values[structure] = []
            diffusion_errors[structure] = []

            for node in nodes:
                diff_data = node.diffusion[structure]
                diffusion_values[structure].append(diff_data["mean"])
                diffusion_errors[structure].append(diff_data["std"])

            # Calculate relative differences (percentage from first node)
            ref_value = diffusion_values[structure][0]
            rel_diffs = [
                100 * (val - ref_value) / ref_value if ref_value != 0 else 0
                for val in diffusion_values[structure]
            ]
            relative_differences[structure] = rel_diffs

        # Get unit from first node, first structure
        unit = nodes[0].diffusion[common_structures[0]]["unit"]

        # Create diffusion coefficient comparison
        diff_fig = create_bar_comparison(
            diffusion_values,
            diffusion_errors,
            labels,
            title="Diffusion Coefficient Comparison",
            ylabel=f"D / {unit}",
            use_plotly=use_plotly,
        )
        figures["diffusion_coefficients"] = diff_fig

        # Create relative difference plot
        rel_diff_fig = create_bar_comparison(
            relative_differences,
            None,
            labels,
            title=f"Relative Differences (vs. {labels[0]})",
            ylabel="Relative Difference / %",
            use_plotly=use_plotly,
        )
        figures["relative_differences"] = rel_diff_fig

        # Create MSD overlay plots for each structure
        for structure in common_structures:
            msd_times = []
            msd_values = []

            # Load MSD data from each node
            for node in nodes:
                try:
                    safe_structure = sanitize_structure_name(structure)
                    msd_file = node.data_path / f"{safe_structure}_msd.h5"
                    dt_file = node.data_path / f"{safe_structure}_dt.h5"

                    if msd_file.exists() and dt_file.exists():
                        # Load using scipp
                        msd = sc.io.load_hdf5(msd_file)
                        dt = sc.io.load_hdf5(dt_file)

                        # Convert time to ps for consistent plotting
                        dt_ps = sc.to_unit(dt, "ps")

                        msd_times.append(dt_ps.values)
                        msd_values.append(msd.values)
                    else:
                        log.warning(f"MSD data not found for {structure} in {node.name}")
                        # Use placeholder data
                        msd_times.append(np.array([0]))
                        msd_values.append(np.array([0]))
                except Exception as e:
                    log.warning(
                        f"Could not load MSD data for {structure} from {node.name}: {e}"
                    )
                    msd_times.append(np.array([0]))
                    msd_values.append(np.array([0]))

            # Only create plot if we have valid data
            if all(len(t) > 1 for t in msd_times):
                msd_fig = create_overlay_plot(
                    msd_times,
                    msd_values,
                    labels,
                    title=f"MSD Comparison: {structure}",
                    xlabel="Time / ps",
                    ylabel="MSD / Å²",
                    use_plotly=use_plotly,
                )
                figures[f"msd_{structure}"] = msd_fig

        return {"figures": figures}

    def _compute_diffusion(
        self,
        molecules: tuple[tuple[int, ...]],
        masses: list[int],
        frames: list[ase.Atoms],
    ) -> DiffusionAnalyzer:
        # effective stride in MD steps between consecutive frames in `frames`
        effective_skip = self.step * self.sampling_rate

        params = {
            "specie": None,
            "time_step": self.time_step * sc.Unit("fs"),
            "step_skip": effective_skip * sc.Unit("dimensionless"),
            "specie_indices": sc.array(
                dims=["particle", "atoms in particle"],
                values=molecules,
                unit=sc.Unit("dimensionless"),
            ),
            "masses": sc.array(dims=["atoms in particle"], values=masses),
            "progress": True,
        }
        if self.dt is not None:
            params["dt"] = sc.arange(
                dim="time interval",
                start=self.dt[0] * sc.Unit("fs"),
                stop=self.dt[1] * sc.Unit("fs"),
                step=self.dt[2] * sc.Unit("fs"),
            )
        diff = DiffusionAnalyzer.from_ase(frames, **params)

        diff.diffusion(self.start_dt * sc.Unit("fs"))
        return diff

    def _save_diff(self, diff: DiffusionAnalyzer, structure: str) -> None:
        safe_structure = sanitize_structure_name(structure)
        diff.msd.save_hdf5(self.data_path / f"{safe_structure}_msd.h5")
        diff.D._to_datagroup().save_hdf5(
            self.data_path / f"{safe_structure}_D.h5"
        )  # this is not being saved!
        diff.dt.save_hdf5(self.data_path / f"{safe_structure}_dt.h5")
        np.save(
            self.data_path / f"{safe_structure}_distributions.npy",
            diff.distributions,
        )

    def _plot(self, diff: DiffusionAnalyzer, structure: str) -> None:
        safe_structure = sanitize_structure_name(structure)
        d_mean = sc.mean(diff.D).value
        d_std = sc.std(diff.D, ddof=1).value

        dt = sc.to_unit(diff.dt, "ps")
        start_dt = self.start_dt * sc.Unit("fs")
        start_dt = sc.to_unit(start_dt, "ps")

        # Define sigma levels
        sigmas = [1, 2, 3]
        alpha = [0.6, 0.4, 0.2]

        # Compute corresponding percentile ranges for each sigma
        credible_intervals = []
        for s in sigmas:
            p_low = 100 * st.norm.cdf(-s)
            p_high = 100 * st.norm.cdf(s)
            credible_intervals.append([p_low, p_high])

        # --- MSD plot ---
        fig, ax = plt.subplots()
        ax.plot(
            dt.values,
            diff.msd.values,
            "k-",
            label=f"D = {d_mean:.2e} ± {d_std:.2e} {diff.D.unit}",
        )
        for i, (ci, s) in enumerate(zip(credible_intervals, sigmas)):
            ax.fill_between(
                dt.values,
                *np.percentile(diff.distributions, ci, axis=1),
                alpha=alpha[i],
                color="#0173B2",
                lw=0,
                label=f"±{s}σ interval",
            )
        ax.set_xlabel(f"Time / {dt.unit}")
        ax.set_ylabel(f"MSD / {diff.msd.unit}")
        ax.axvline(
            start_dt.value,
            c="k",
            ls="--",
            label=f"start_dt = {start_dt.value} {start_dt.unit}",
        )
        ax.legend()
        ax.set_title(f"{structure} MSD credible intervals")
        fig.savefig(
            self.figures_path / f"{safe_structure}_msd.png", dpi=300, bbox_inches="tight"
        )

        # --- Histogram plot ---
        fig, ax = plt.subplots()
        ax.hist(
            diff.D.values,
            density=True,
            bins=50,
            color="lightblue",
            edgecolor="k",
        )
        ax.axvline(
            d_mean,
            c="red",
            ls="--",
            label=f"MAP D = {d_mean:.2e} ± {d_std:.2e} {diff.D.unit}",
        )

        # Add σ-based intervals as vertical lines
        for i, (ci, s) in enumerate(zip(credible_intervals, sigmas)):
            low, high = np.percentile(diff.D.values, ci)
            ax.axvline(
                low,
                c="orange",
                ls="--",
                alpha=alpha[i],
                label=f"±{s}σ interval",
            )
            ax.axvline(
                high,
                c="orange",
                ls="--",
                alpha=alpha[i],
            )

        ax.set_xlabel(f"Diffusion Coefficient D* / {diff.D.unit}")
        ax.set_ylabel(f"p(D*) / {diff.D.unit}")
        ax.legend()
        ax.set_title(f"{structure} Diffusion Coefficient Distribution")
        fig.savefig(
            self.figures_path / f"{safe_structure}_D_histogram.png",
            dpi=300,
            bbox_inches="tight",
        )

    def run(self):
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.figures_path.mkdir(parents=True, exist_ok=True)
        self.diffusion = {}

        # --- Data Loading ---
        io = self.data
        if isinstance(io, znh5md.IO):
            io.include = ["position", "box"]

        frames = io[self.start : self.stop : self.step]
        graph = rdkit2ase.ase2networkx(frames[0], suggestions=self.structures)
        molecules: dict[str, tuple[tuple[int, ...]]] = {}
        masses: dict[str, list[int]] = {}
        if self.structures is None:
            symbol_map = frames[0].symbols.indices()
            for k, v in symbol_map.items():
                molecules[k] = tuple((int(i),) for i in v)
                masses[k] = [frames[0].get_masses()[v[0]]]
        else:
            for structure in self.structures:
                matches = rdkit2ase.match_substructure(
                    rdkit2ase.networkx2ase(graph), smiles=structure
                )
                if not matches:
                    log.warning(f"No matches found for structure {structure}")
                molecules[structure] = matches
                masses[structure] = list(frames[0].get_masses()[list(matches[0])])

        for structure in molecules:
            print(f"Calculating diffusion for {structure}")
            diff = self._compute_diffusion(
                molecules[structure], masses[structure], frames
            )
            self._save_diff(diff, structure)
            self._plot(diff, structure)

            self.diffusion[structure] = {
                "mean": float(sc.mean(diff.D).value),
                "std": float(sc.std(diff.D, ddof=1).value),
                "var": float(sc.var(diff.D, ddof=1).value),
                "occurrences": len(molecules[structure]),
                "unit": str(diff.D.unit),
                "box": frames[0].cell.array.tolist(),
            }

    @property
    def frames(self) -> znh5md.IO:
        """Return trajectory frames as a list of ASE Atoms objects."""
        if self.file is not None:
            file_factory = make_hdf5_file_opener(self, self.file)
            return znh5md.IO(file_factory=file_factory)
        elif self.data is not None:
            return self.data
        else:
            raise ValueError("Either 'file' or 'data' must be provided.")
