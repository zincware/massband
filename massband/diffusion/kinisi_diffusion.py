import logging
import pickle
from pathlib import Path
from typing import Optional, TypedDict, Union

import numpy as np
import plotly.graph_objects as go
import rdkit2ase
import scipp as sc
import znh5md
import zntrack
from kinisi.analyze import DiffusionAnalyzer

from massband.abc import ComparisonResults
from massband.kinisi import KinisiPlotData
from massband.plotting.kinisi import PlottingConfig, plot_kinisi_results

import numpy as np
from ase.io import read
import matplotlib.pyplot as plt
from kinisi.analyze import DiffusionAnalyzer
import scipp as sc

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
    structures: list[str] = zntrack.params()
    start: int = zntrack.params(0)
    stop: int | None = zntrack.params(None)
    step: int = zntrack.params(1)

    time_step: float = zntrack.params()  # in fs
    sampling_rate: int = zntrack.params()  # in number of frames

    data_path: Path = zntrack.outs_path(zntrack.nwd / "diffusion_data")
    dt: tuple[float, float, float] | None = zntrack.params()
    start_dt: float = zntrack.params()  # in fs

    def run(self):
        self.data_path.mkdir(parents=True, exist_ok=True)
        io = znh5md.IO(self.file, include=["position", "box"])
        frames = io[self.start : self.stop : self.step]
        graph = rdkit2ase.ase2networkx(frames[0], suggestions=self.structures)
        molecules: dict[str, tuple[tuple[int, ...]]] = {}
        masses: dict[str, list[int]] = {}
        for structure in self.structures:
            matches = rdkit2ase.match_substructure(
                rdkit2ase.networkx2ase(graph), smiles=structure
            )
            if not matches:
                log.warning(f"No matches found for structure {structure}")
            molecules[structure] = matches
            masses[structure] = list(frames[0].get_masses()[list(matches[0])])

        for structure in self.structures:
            print(f"Calculating diffusion for {structure}.")
            params = {
                "specie": None,
                "time_step": self.time_step * sc.Unit("fs"),
                "step_skip": self.step * self.sampling_rate * sc.Unit("dimensionless"),
                "specie_indices": sc.array(
                    dims=["particle", "atoms in particle"],
                    values= molecules[structure],
                    unit=sc.Unit("dimensionless"),
                ),
                "masses": sc.array(dims=["atoms in particle"], values=masses[structure]),
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

            diff.diffusion(self.start_dt * sc.Unit('fs'))


            credible_intervals = [[16, 84], [2.5, 97.5], [0.15, 99.85]]
            alpha = [0.6, 0.4, 0.2]

            fig, ax = plt.subplots()
            ax.plot(diff.dt.values, diff.msd.values, 'k-')
            for i, ci in enumerate(credible_intervals):
                ax.fill_between(diff.dt.values,
                                *np.percentile(diff.distributions, ci, axis=1),
                                alpha=alpha[i],
                                color='#0173B2',
                                lw=0)
            ax.set_xlabel(f'Time / {diff.dt.unit}')
            ax.set_ylabel(f'MSD / {diff.msd.unit}')
            fig.savefig(self.data_path / f"{structure}_msd.png", dpi=300)
        

