import logging
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TypedDict, Union

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import zntrack
from ase import Atoms
# from kinisi.analyze import DiffusionAnalyzer
from kinisi.parser import ASEParser
from kinisi.diffusion import MSDBootstrap
from tqdm import tqdm

from massband.abc import ComparisonResults
from massband.dataloader import SpeciesBatchedLoader

log = logging.getLogger(__name__)


class DiffusionResults(TypedDict):
    diffusion_coefficient: float
    std: float
    credible_interval_68: list[float]
    credible_interval_95: list[float]
    asymmetric_uncertainty: list[float]
    occurrences: int


@dataclass
class DiffusionPlotData:
    structure: str
    dt: np.ndarray
    msd: np.ndarray
    msd_std: np.ndarray
    distribution: np.ndarray
    D_samples: np.ndarray
    D_n: float


class KinisiSelfDiffusion(zntrack.Node):
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

    data_path: Path = zntrack.outs_path(zntrack.nwd / "diffusion_data")

    def process_species_batches(self):
        """Iterate through SpeciesBatchedLoader and yield complete species data."""
        # Use SpeciesBatchedLoader to get batches
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
                # Accumulate positions for each species
                data[species_name].append(positions)

        for species_name, positions_list in data.items():
            # Concatenate all positions for this species
            combined_positions = np.concatenate(positions_list, axis=1)
            log.info(
                f"Yielding complete data for species {species_name} with shape {combined_positions.shape}"
            )
            yield species_name, combined_positions

        # # Iterate through all batches
        # for batch_data, _, _ in tqdm(loader):
        #     # TODO: This mustn't work because the species are not guaranteed to be in the same order
        #     for species_name, positions in batch_data.items():
        #         # Check if we've moved to a new species
        #         if current_species is None:
        #             # First species
        #             current_species = species_name
        #             accumulated_positions = [positions]
        #         elif species_name != current_species:
        #             # New species detected - yield the previous species data
        #             if accumulated_positions:
        #                 combined_positions = np.concatenate(accumulated_positions, axis=1)
        #                 log.info(f"Yielding complete data for species {current_species} with shape {combined_positions.shape}")
        #                 yield current_species, combined_positions

        #             # Start accumulating for new species
        #             current_species = species_name
        #             accumulated_positions = [positions]
        #         else:
        #             # Same species - accumulate positions
        #             # different shapes with com / not com
        #             accumulated_positions.append(positions)

        # # Yield the last species if we have data
        # if current_species is not None and accumulated_positions:
        #     combined_positions = np.concatenate(accumulated_positions, axis=0)
        #     log.info(f"Yielding final species {current_species} with shape {combined_positions.shape}")
        #     yield current_species, combined_positions

    def run(self):
        np.random.seed(self.seed)
        self.results = {}
        self.data_path.mkdir(exist_ok=True, parents=True)

        # Account for step in the effective time step
        effective_time_step = self.time_step / 1000 * self.step

        # Process each species individually
        for species_name, positions in self.process_species_batches():
            log.info(f"Processing diffusion analysis for {species_name}")

            # positions shape: (n_frames, n_molecules, 3)
            n_frames, n_molecules, _ = positions.shape
            # Create ASE Atoms objects for each frame
            # Since we're working with COM positions, we treat each as a single atom
            frames = []
            for frame_idx in range(n_frames):
                frame_positions = positions[frame_idx]  # Shape: (n_molecules, 3)

                # Create atoms object with dummy atomic numbers (e.g., all as hydrogen)
                atoms = Atoms(
                    # symbols=["H"] * n_molecules,
                    positions=frame_positions,
                    pbc=False,
                    cell=[100000, 1000000, 100000],  # required for kinisi to work
                )
                frames.append(atoms)

            # For COM positions, each "molecule" is treated as a single entity
            occurrences = n_molecules

            try:
                # diff = DiffusionAnalyzer.from_ase(
                #     frames,
                #     parser_params={
                #         "specie": "H",
                #         "time_step": effective_time_step,
                #         "step_skip": self.sampling_rate,
                #         "progress": True,
                #     },
                #     uncertainty_params={"progress": True},
                # )
                # diff.diffusion(self.start_dt, {"progress": True})
                diff = ASEParser(
                    atoms=frames, specie="X", time_step=effective_time_step, step_skip=self.sampling_rate,
                )
                diff = MSDBootstrap(diff.delta_t, diff.disp_3d, diff._n_o)
                diff.diffusion(start_dt=self.start_dt)
                distribution = diff.gradient.samples * diff.dt[:, np.newaxis] + diff.intercept.samples

                result = DiffusionPlotData(
                    structure=species_name,
                    dt=np.asarray(diff.dt),
                    msd=np.asarray(diff.n),
                    msd_std=np.asarray(diff.s),
                    distribution=np.asarray(distribution),
                    D_samples=np.asarray(diff.D.samples),
                    D_n=float(diff.D.n),
                )

                with open(self.data_path / f"{species_name}.pkl", "wb") as f:
                    pickle.dump(result, f)

                # Compute uncertainty statistics from D_samples
                D_mean = result.D_n
                D_std = np.std(result.D_samples)
                ci68 = np.percentile(result.D_samples, [16, 84])
                ci95 = np.percentile(result.D_samples, [2.5, 97.5])

                # Optional: asymmetric error bars
                uncertainty_low = D_mean - ci68[0]
                uncertainty_high = ci68[1] - D_mean

                # Store in results
                self.results[species_name] = {
                    "diffusion_coefficient": result.D_n,
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
        credible_intervals = [[16, 84], [2.5, 97.5], [0.15, 99.85]]
        alpha = [0.6, 0.4, 0.2]

        for pkl_path in self.data_path.glob("*.pkl"):
            with open(pkl_path, "rb") as f:
                data: DiffusionPlotData = pickle.load(f)

            # MSD with std
            fig, ax = plt.subplots()
            ax.errorbar(data.dt, data.msd, data.msd_std)
            ax.set_ylabel("MSD/Å$^2$")
            ax.set_xlabel(r"$\Delta t$/ps")
            ax.set_title(f"{data.structure} MSD with std")
            fig.savefig(self.data_path / f"{data.structure}_msd_std.png", dpi=300)
            plt.close(fig)

            # MSD with credible intervals
            fig, ax = plt.subplots()
            ax.plot(data.dt, data.msd, "k-")
            for i, ci in enumerate(credible_intervals):
                low, high = np.percentile(data.distribution, ci, axis=1)
                ax.fill_between(data.dt, low, high, alpha=alpha[i], color="#0173B2", lw=0)
            # TODO: save start_dt in pickle as well?
            ax.axvline(self.start_dt, c="k", ls="--")
            ax.set_ylabel("MSD/Å$^2$")
            ax.set_xlabel(r"$\Delta t$/ps")
            ax.set_title(f"{data.structure} MSD credible intervals")
            fig.savefig(
                self.data_path / f"{data.structure}_credible_intervals.png", dpi=300
            )
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.hist(
                data.D_samples, density=True, bins=50, color="lightblue", edgecolor="k"
            )
            ax.axvline(data.D_n, c="red", ls="--", label="MAP (D_n)")
            ax.axvline(
                self.results[data.structure]["credible_interval_68"][0],
                c="blue",
                ls=":",
                label="68% CI",
            )
            ax.axvline(
                self.results[data.structure]["credible_interval_68"][1], c="blue", ls=":"
            )

            ax.set_xlabel("$D$/cm$^2$s$^{-1}$")
            ax.set_ylabel("$p(D)$/cm$^2$s$^{-1}$")
            ax.set_title(f"{data.structure} Diffusion Histogram")
            ax.legend()

            # Annotate uncertainty
            textstr = "\n".join(
                (
                    f"Mean: {self.results[data.structure]['diffusion_coefficient']:.3e}",
                    f"Std: ±{self.results[data.structure]['std']:.3e}",
                    f"68% CI: [{self.results[data.structure]['credible_interval_68'][0]:.3e}, {self.results[data.structure]['credible_interval_68'][1]:.3e}]",
                    f"95% CI: [{self.results[data.structure]['credible_interval_95'][0]:.3e}, {self.results[data.structure]['credible_interval_95'][1]:.3e}]",
                    f"Asymmetric Uncertainty: [{self.results[data.structure]['asymmetric_uncertainty'][0]:.3e}, {self.results[data.structure]['asymmetric_uncertainty'][1]:.3e}]",
                )
            )
            ax.text(
                0.95,
                0.95,
                textstr,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
                fontsize=8,
            )

            fig.savefig(self.data_path / f"{data.structure}_hist.png", dpi=300)
            plt.close(fig)

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
        # Find common structures across all nodes by looking at the keys of the results dict.
        all_structures = [set(node.results.keys()) for node in nodes if node.results]
        if not all_structures:
            return {"frames": [], "figures": {}}

        common_structures = set.intersection(*all_structures)

        for structure in common_structures:
            # --- MSD Comparison Plot ---
            msd_fig = go.Figure()
            # --- Histogram Comparison Plot ---
            hist_fig = go.Figure()

            for node in nodes:
                # Check if the node has data for the current structure
                if not node.results or structure not in node.results:
                    continue

                # Load the pickled data for this node and structure
                data_file = node.data_path / f"{structure}.pkl"
                if not data_file.exists():
                    continue
                with open(data_file, "rb") as f:
                    data: DiffusionPlotData = pickle.load(f)

                # Add MSD trace to the MSD comparison plot
                msd_fig.add_trace(
                    go.Scatter(
                        x=data.dt,
                        y=data.msd,
                        mode="lines",
                        name=f"{node.name}",
                    )
                )

                # Add histogram trace to the histogram comparison plot
                hist_fig.add_trace(
                    go.Histogram(
                        x=data.D_samples,
                        name=f"{node.name}",
                        opacity=0.7,
                    )
                )

            # Style the MSD plot
            msd_fig.update_layout(
                title_text=f"MSD Comparison for: {structure}",
                xaxis_title_text=r"$\Delta t$/ps",
                yaxis_title_text="MSD/Å²",
                legend_title_text="Compared Runs",
            )
            figures[f"msd_comparison_{structure}"] = msd_fig

            # Style the histogram plot
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
