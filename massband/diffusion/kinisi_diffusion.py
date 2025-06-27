import logging
import pickle
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import zntrack
from ase import Atoms
from collections import defaultdict
from functools import partial
from jax import vmap
from massband.utils import unwrap_positions
from znh5md import IO
import rdkit2ase
from kinisi.analyze import DiffusionAnalyzer

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
        data = self.get_data()
        unwrap_positions_vmap = vmap(
            partial(unwrap_positions, cells=data["cells"], inv_cells=data["inv_cells"]),
            in_axes=(1,),
        )
        positions_unwrapped = unwrap_positions_vmap(data["positions"])
        log.info(f"Unwrapped positions shape: {positions_unwrapped.shape}")

        substructures = defaultdict(list)
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

        self.data_path.mkdir(exist_ok=True)

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
                ax.fill_between(
                    data.dt, low, high, alpha=alpha[i], color="#0173B2", lw=0
                )
            # TODO: save start_dt in pickle as well?
            ax.axvline(self.start_dt, c="k", ls="--")
            ax.set_ylabel("MSD/Å$^2$")
            ax.set_xlabel(r"$\Delta t$/ps")
            ax.set_title(f"{data.structure} MSD credible intervals")
            fig.savefig(
                self.data_path / f"{data.structure}_credible_intervals.png", dpi=300
            )
            plt.close(fig)

            # Histogram of diffusion coefficients
            fig, ax = plt.subplots()
            ax.hist(data.D_samples, density=True, bins=50)
            ax.axvline(data.D_n, c="k")
            ax.set_xlabel("$D$/cm$^2$s$^{-1}$")
            ax.set_ylabel("$p(D)$/cm$^2$s$^{-1}$")
            ax.set_title(f"{data.structure} Diffusion Histogram")
            fig.savefig(self.data_path / f"{data.structure}_hist.png", dpi=300)
            plt.close(fig)
