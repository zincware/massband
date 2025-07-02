import logging
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import rdkit2ase
import zntrack
from ase import Atoms
from kinisi.analyze import DiffusionAnalyzer
from znh5md import IO

from massband.abc import ComparisonResults
from massband.utils import unwrap_positions

log = logging.getLogger(__name__)


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
    results: dict = zntrack.metrics()
    seed: int = zntrack.params(42)

    data_path: Path = zntrack.outs_path(zntrack.nwd / "diffusion_data")

    def get_data(self):
        io = IO(self.file, variable_shape=False, include=["position", "box"])
        frames: list[Atoms] = io[:]
        positions = jnp.stack([atoms.positions for atoms in frames])
        cells = jnp.stack([atoms.cell[:] for atoms in frames])
        inv_cells = jnp.linalg.inv(cells)

        return {
            "positions": positions,
            "cells": cells,
            "inv_cells": inv_cells,
            "frames": frames,
        }

    def run(self):
        # TODO: need to make sure that in the first frame, all molecules are fully wrapped!!
        np.random.seed(self.seed)
        self.results = {}
        data = self.get_data()
        positions_unwrapped = unwrap_positions(
            data["positions"], data["cells"], data["inv_cells"]
        )
        log.info(f"Unwrapped positions shape: {positions_unwrapped.shape}")

        substructures = defaultdict(list)
        # TODO: use com stuff
        if self.structures:
            log.info(f"Searching for substructures in {len(self.structures)} patterns")
            for structure in self.structures:
                indices = rdkit2ase.match_substructure(
                    data["frames"][0],
                    smiles=structure,
                    suggestions=self.structures,
                )
                if indices:
                    substructures[structure].extend(indices)
                    log.info(f"Found {len(indices)} matches for {structure}")

        self.data_path.mkdir(exist_ok=True, parents=True)

        for structure, indices in substructures.items():
            flat_indices = [i for sublist in indices for i in sublist]
            sub_frames = [atoms[flat_indices] for atoms in data["frames"]]
            specie_indices = rdkit2ase.match_substructure(
                sub_frames[0],
                smiles=structure,
                suggestions=self.structures,
            )
            masses = sub_frames[0][specie_indices[0]].get_masses().tolist()

            diff = DiffusionAnalyzer.from_ase(
                sub_frames,
                parser_params={
                    "specie": None,
                    "time_step": self.time_step / 1000,
                    "step_skip": self.sampling_rate,
                    "specie_indices": specie_indices,
                    "masses": masses,
                    "progress": True,
                },
                uncertainty_params={"progress": True},
            )
            diff.diffusion(self.start_dt, {"progress": True})

            result = DiffusionPlotData(
                structure=structure,
                dt=np.asarray(diff.dt),
                msd=np.asarray(diff.msd),
                msd_std=np.asarray(diff.msd_std),
                distribution=np.asarray(diff.distribution),
                D_samples=np.asarray(diff.D.samples),
                D_n=float(diff.D.n),
            )

            with open(self.data_path / f"{structure}.pkl", "wb") as f:
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
            self.results[structure] = {
                "diffusion_coefficient": result.D_n,
                "std": D_std,
                "credible_interval_68": ci68.tolist(),
                "credible_interval_95": ci95.tolist(),
                "asymmetric_uncertainty": [uncertainty_low, uncertainty_high],
            }

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
