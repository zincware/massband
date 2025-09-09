import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Optional, TypedDict, Union

import numpy as np
import plotly.graph_objects as go
import zntrack
from ase import Atoms
from kinisi.diffusion import MSDBootstrap
from kinisi.parser import ASEParser
from tqdm import tqdm

from massband.abc import ComparisonResults
from massband.dataloader import SpeciesBatchedLoader
from massband.kinisi import KinisiPlotData
from massband.plotting.kinisi import PlottingConfig, plot_kinisi_results

log = logging.getLogger(__name__)


class DiffusionResults(TypedDict):
    diffusion_coefficient: float
    std: float
    credible_interval_68: list[float]
    credible_interval_95: list[float]
    asymmetric_uncertainty: list[float]
    occurrences: int


class KinisiSelfDiffusion(zntrack.Node):
    """
    Self-diffusion coefficient calculation using kinisi library.

    Parameters
    ----------
    file : Union[str, Path]
        Path to H5MD trajectory file.
    sampling_rate : int
        Number of steps to skip between frames for analysis.
    time_step : float
        Simulation time step in femtoseconds.
    structures : Optional[list[str]], optional
        List of SMILES strings for molecular structures to analyze.
    start_dt : float, default 50
        Start time for diffusion analysis in picoseconds.
    seed : int, default 42
        Random seed for bootstrap sampling.
    batch_size : int, default 64
        Batch size for trajectory processing.
    start : int, default 0
        Starting frame index (in steps).
    stop : Optional[int], optional
        Stopping frame index (in steps).
    step : int, default 1
        Frame step size (in steps).
    memory_limit : float, default 16
        Memory limit in GB for kinisi processing.

    Examples
    --------
    >>> with project:
    ...     diff = massband.KinisiSelfDiffusion(
    ...         file=ec_emc,
    ...         time_step=0.5,
    ...         sampling_rate=1000,
    ...         structures=["C1COC(=O)O1", "CCOC(=O)OC"],
    ...         start_dt=5000,
    ...     )
    >>> project.repro()
    >>> diff.results["C1COC(=O)O1"].keys()
    dict_keys(['diffusion_coefficient', 'std', 'credible_interval_68', 'credible_interval_95', 'asymmetric_uncertainty', 'occurrences'])

    """

    file: Union[str, Path] = zntrack.deps_path()
    sampling_rate: int = zntrack.params()
    time_step: float = zntrack.params()
    structures: Optional[list[str]] = zntrack.params(None)
    start_dt: float = zntrack.params(50)  # in ps
    results: dict[str, DiffusionResults] = zntrack.metrics()
    seed: int = zntrack.params(42)
    batch_size: int = zntrack.params(64)
    start: int = zntrack.params(0)
    stop: Optional[int] = zntrack.params(None)
    step: int = zntrack.params(1)
    memory_limit: float = zntrack.params(16)  # 16 GB

    data_path: Path = zntrack.outs_path(zntrack.nwd / "diffusion_data")

    def process_species_batches(self):
        """Iterate through SpeciesBatchedLoader and yield complete species data."""
        loader = SpeciesBatchedLoader(
            file=self.file,
            structures=self.structures,
            batch_size=self.batch_size,
            wrap=False,
            com=True,  # Use center of mass for molecular structures
            memory=True,  # Don't load all into memory
            start=self.start,
            stop=self.stop,
            step=self.step,
        )

        results = list(tqdm(loader))
        data = defaultdict(list)
        for batch_output in results:
            batch_data = batch_output["position"]
            for species_name, positions in batch_data.items():
                data[species_name].append(positions)

        for species_name, positions_list in data.items():
            combined_positions = np.concatenate(positions_list, axis=1)
            log.info(
                f"Yielding complete data for species {species_name} with shape {combined_positions.shape}"
            )
            yield species_name, combined_positions

    def run(self):
        np.random.seed(self.seed)
        self.results = {}
        self.data_path.mkdir(exist_ok=True, parents=True)

        effective_time_step = self.time_step / 1000 * self.step

        for species_name, positions in self.process_species_batches():
            log.info(f"Processing diffusion analysis for {species_name}")

            n_frames, n_molecules, _ = positions.shape
            frames = []
            for frame_idx in range(n_frames):
                frame_positions = positions[frame_idx]
                atoms = Atoms(
                    positions=frame_positions,
                    pbc=False,
                    cell=[100000, 1000000, 100000],
                )
                frames.append(atoms)

            occurrences = n_molecules

            try:
                diff = ASEParser(
                    atoms=frames,
                    specie="X",
                    time_step=effective_time_step,
                    step_skip=self.sampling_rate,
                    memory_limit=self.memory_limit,
                )
                diff = MSDBootstrap(diff.delta_t, diff.disp_3d, diff._n_o)
                diff.diffusion(start_dt=self.start_dt)
                distribution = (
                    diff.gradient.samples * diff.dt[:, np.newaxis]
                    + diff.intercept.samples
                )

                plot_data = KinisiPlotData(
                    structure=species_name,
                    dt=np.asarray(diff.dt),
                    displacement=np.asarray(diff.n),
                    displacement_std=np.asarray(diff.s),
                    distribution=np.asarray(distribution),
                    samples=np.asarray(diff.D.samples),
                    mean_value=float(diff.D.n),
                    start_dt=self.start_dt,
                )

                with open(self.data_path / f"{species_name}.pkl", "wb") as f:
                    pickle.dump(plot_data, f)

                D_mean = plot_data.mean_value
                D_std = np.std(plot_data.samples)
                ci68 = np.percentile(plot_data.samples, [16, 84])
                ci95 = np.percentile(plot_data.samples, [2.5, 97.5])

                uncertainty_low = D_mean - ci68[0]
                uncertainty_high = ci68[1] - D_mean

                self.results[species_name] = {
                    "diffusion_coefficient": D_mean,
                    "std": D_std,
                    "credible_interval_68": ci68.tolist(),
                    "credible_interval_95": ci95.tolist(),
                    "asymmetric_uncertainty": [uncertainty_low, uncertainty_high],
                    "occurrences": occurrences,
                }

                log.info(
                    f"Completed diffusion analysis for {species_name}: D = {D_mean:.3e} cm²/s"
                )

            except Exception as e:
                log.error(f"Failed to process species {species_name}: {e}")
                continue

        self.plot()

    def plot(self):
        for pkl_path in self.data_path.glob("*.pkl"):
            with open(pkl_path, "rb") as f:
                data: KinisiPlotData = pickle.load(f)

            config = PlottingConfig(
                displacement_label="MSD",
                displacement_unit="Å$^2$",
                value_label="D",
                value_unit="cm$^2$s$^{-1}$",
                msd_title=f"{data.structure} MSD with std",
                msd_filename=f"{data.structure}_msd_std.png",
                ci_title=f"{data.structure} MSD credible intervals",
                ci_filename=f"{data.structure}_credible_intervals.png",
                hist_title=f"{data.structure} Diffusion Histogram",
                hist_filename=f"{data.structure}_hist.png",
                hist_label="MAP (D_n)",
            )
            plot_kinisi_results(data, self.data_path, config, self.results)

    @classmethod
    def compare(cls, *nodes: "KinisiSelfDiffusion") -> ComparisonResults:
        """
        Compare diffusion results from multiple runs.

        This method generates two types of plots for each common structure found
        across the provided nodes:
        1.  An overlay of the Mean Squared Displacement (MSD) curves.
        2.  An overlay of the diffusion coefficient histograms.
        """
        figures = {}
        all_structures = [set(node.results.keys()) for node in nodes if node.results]
        if not all_structures:
            return {"frames": [], "figures": {}}

        common_structures = set.intersection(*all_structures)

        for structure in common_structures:
            msd_fig = go.Figure()
            hist_fig = go.Figure()

            for node in nodes:
                if not node.results or structure not in node.results:
                    continue

                data_file = node.data_path / f"{structure}.pkl"
                if not data_file.exists():
                    continue
                with open(data_file, "rb") as f:
                    data: KinisiPlotData = pickle.load(f)

                msd_fig.add_trace(
                    go.Scatter(
                        x=data.dt,
                        y=data.displacement,
                        mode="lines",
                        name=f"{node.name}",
                    )
                )

                hist_fig.add_trace(
                    go.Histogram(
                        x=data.samples,
                        name=f"{node.name}",
                        opacity=0.7,
                    )
                )

            msd_fig.update_layout(
                title_text=f"MSD Comparison for: {structure}",
                xaxis_title_text=r"$\Delta t$/ps",
                yaxis_title_text="MSD/Å²",
                legend_title_text="Compared Runs",
            )
            figures[f"msd_comparison_{structure}"] = msd_fig

            hist_fig.update_layout(
                barmode="overlay",
                title_text=f"Diffusion Coefficient Comparison for: {structure}",
                xaxis_title_text="$D$/cm²s⁻¹",
                yaxis_title_text="Count",
                legend_title_text="Compared Runs",
            )
            hist_fig.update_traces(opacity=0.6)
            figures[f"hist_comparison_{structure}"] = hist_fig

        return {"frames": [], "figures": figures}
