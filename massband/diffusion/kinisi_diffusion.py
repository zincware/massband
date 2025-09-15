import logging
from pathlib import Path
from typing import Union

import ase
import matplotlib.pyplot as plt
import numpy as np
import rdkit2ase
import scipp as sc
import scipy.stats as st
import znh5md
import zntrack
from kinisi.analyze import DiffusionAnalyzer
import typing as t
import os
import h5py
import contextlib

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
    file : Union[str, Path]
        Path to the trajectory file in h5md format.
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

    file: Union[str, Path] = zntrack.deps_path()
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

    diffusion: dict[str, float] = zntrack.metrics()

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
        diff.msd.save_hdf5(self.data_path / f"{structure}_msd.h5")
        diff.D._to_datagroup().save_hdf5(
            self.data_path / f"{structure}_D.h5"
        )  # this is not being saved!
        diff.dt.save_hdf5(self.data_path / f"{structure}_dt.h5")
        np.save(
            self.data_path / f"{structure}_distributions.npy",
            diff.distributions,
        )


    def _plot(self, diff: DiffusionAnalyzer, structure: str) -> None:
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
            self.figures_path / f"{structure}_msd.png", dpi=300, bbox_inches="tight"
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
            self.figures_path / f"{structure}_D_histogram.png",
            dpi=300,
            bbox_inches="tight",
        )

    def run(self):
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.figures_path.mkdir(parents=True, exist_ok=True)
        self.diffusion = {}
        io = znh5md.IO(self.file, include=["position", "box"])
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
            }


    @property
    def frames(self) -> znh5md.IO:
        """Return trajectory frames as a list of ASE Atoms objects."""
        file_factory = make_hdf5_file_opener(self, self.file)
        return znh5md.IO(file_factory=file_factory)