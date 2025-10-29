import logging
from pathlib import Path

import ase
import matplotlib.pyplot as plt
import numpy as np
import rdkit2ase
import scipp as sc
import znh5md
import zntrack
from kinisi.analyze import ConductivityAnalyzer
from rdkit import Chem

from massband.utils import sanitize_structure_name

log = logging.getLogger(__name__)


class KinisiEinsteinHelfandIonicConductivity(zntrack.Node):
    """Compute ionic conductivity using the kinisi library.

    Analyzes molecular dynamics trajectories to calculate system-wide ionic conductivity
    and mean squared charge displacements for all ionic species combined.

    Parameters
    ----------
    data: znh5md.IO | list[ase.Atoms] | None, default None
        znh5md.IO object for trajectory data, as an alternative to 'file'.
    structures : list[str]
        List of SMILES strings representing ionic structures in the system.
        Must include both cations and anions (e.g., ["[Li+]", "F[P-](F)(F)(F)(F)F"]).
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
        Minimum time interval for conductivity coefficient fitting in femtoseconds.
    temperature : float
        System temperature in Kelvin for conductivity calculation.

    Attributes
    ----------
    data_path : Path
        Output directory for data files.
    figures_path : Path
        Output directory for plots.
    conductivity : dict[str, dict[str, float]]
        Dictionary mapping "total" to the total system-wide conductivity statistics.
        Contains mean, std, var, and unit under the "total" key.

    Examples
    --------
    >>> with project:
    ...     cond = massband.KinisiEinsteinHelfandIonicConductivity(
    ...         file=ec_emc,
    ...         time_step=0.5,
    ...         sampling_rate=1000,
    ...         structures=["[Li+]", "F[P-](F)(F)(F)(F)F"],
    ...         start_dt=500_000,
    ...         temperature=300.0,
    ...         step=1000,
    ...     )
    >>> project.repro()
    >>> cond.conductivity.keys()
    dict_keys(['total'])
    >>> cond.conductivity["total"].keys()
    dict_keys(['mean', 'std', 'var', 'unit'])

    References
    ----------
    .. [1] https://kinisi.readthedocs.io/en/stable/
    """

    data: znh5md.IO | list[ase.Atoms] = zntrack.deps()
    structures: list[str] = zntrack.params()
    start: int = zntrack.params(0)
    stop: int | None = zntrack.params(None)
    step: int = zntrack.params(1)

    time_step: float = zntrack.params()  # in fs
    sampling_rate: int = zntrack.params()  # in number of frames

    data_path: Path = zntrack.outs_path(zntrack.nwd / "data")
    figures_path: Path = zntrack.outs_path(zntrack.nwd / "figures")
    dt: tuple[float, float, float] | None = zntrack.params(None)
    start_dt: float = zntrack.params()  # in fs
    temperature: float = zntrack.params()  # in K

    conductivity: dict[str, dict[str, float | str]] = zntrack.metrics()

    def run(self):
        from kinisi import Species

        self.data_path.mkdir(exist_ok=True, parents=True)
        self.figures_path.mkdir(exist_ok=True, parents=True)

        # --- Data Loading ---
        io = self.data
        if isinstance(io, znh5md.IO):
            io.include = ["position", "box"]
        frames = io[self.start : self.stop : self.step]

        graph = rdkit2ase.ase2networkx(frames[0], suggestions=self.structures)
        molecules: dict[str, tuple[tuple[int, ...]]] = {}
        masses: dict[str, list[int]] = {}
        charge: dict[str, int] = {}

        species = []

        for structure in self.structures:
            matches = rdkit2ase.match_substructure(
                rdkit2ase.networkx2ase(graph), smiles=structure
            )
            if not matches:
                raise ValueError(f"No matches found for structure: {structure}")
            molecules[structure] = matches
            masses[structure] = list(frames[0].get_masses()[list(matches[0])])
            mol = Chem.MolFromSmiles(structure)
            if mol is not None:
                charge[structure] = Chem.GetFormalCharge(mol)

            species.append(
                Species(
                    indices=[list(x) for x in matches],
                    masses=list(frames[0].get_masses()[list(matches[0])]),
                    charge=Chem.GetFormalCharge(mol),
                )
            )

        params = {
            "specie": species,
            "time_step": self.time_step * sc.Unit("fs"),
            "step_skip": self.sampling_rate * self.step * sc.Unit("dimensionless"),
            "progress": True,
        }
        if self.dt is not None:
            params["dt"] = sc.arange(
                dim="time interval",
                start=self.dt[0] * sc.Unit("fs"),
                stop=self.dt[1] * sc.Unit("fs"),
                step=self.dt[2] * sc.Unit("fs"),
            )

        cond = ConductivityAnalyzer.from_ase(
            trajectory=frames,
            **params,
        )
        start_dt = self.start_dt * sc.Unit("fs")
        cond.conductivity(
            start_dt=start_dt,
            temperature=self.temperature * sc.Unit("K"),
        )

        credible_intervals = [[16, 84], [2.5, 97.5], [0.15, 99.85]]
        alpha = [0.6, 0.4, 0.2]

        fig, ax = plt.subplots()
        ax.plot(cond.dt.values, cond.mscd.values, "k-")
        for i, ci in enumerate(credible_intervals):
            ax.fill_between(
                cond.dt.values,
                *np.percentile(cond.distributions, ci, axis=1),
                alpha=alpha[i],
                color="#0173B2",
                lw=0,
            )
        ax.axvline(
            start_dt.value,
            c="k",
            ls="--",
            label=f"start_dt = {start_dt.value} {start_dt.unit}",
        )

        ax.set_xlabel(f"Time / {cond.dt.unit}")
        ax.set_ylabel(f"mscd / {cond.mscd.unit}")

        fig.savefig(self.figures_path / "mscd.png", dpi=300, bbox_inches="tight")

        fig, ax = plt.subplots()

        new_sigma = sc.to_unit(cond.sigma, "S/m")

        ax.hist(new_sigma.values, density=True)
        ax.axvline(sc.mean(new_sigma).value, c="k")
        ax.set_xlabel(f"D* / [{new_sigma.unit}]")
        ax.set_ylabel(f"p(D*) / [{(1 / new_sigma.unit).unit}]")
        fig.savefig(
            self.figures_path / "sigma_distribution.png", dpi=300, bbox_inches="tight"
        )
        self.conductivity = {
            "total": {
                "mean": float(sc.mean(new_sigma).value),
                "std": float(sc.std(new_sigma, ddof=1).value),
                "var": float(sc.var(new_sigma, ddof=1).value),
                "unit": str(new_sigma.unit),
            }
        }

        safe_structure = sanitize_structure_name(structure)
        cond.mscd.save_hdf5(self.data_path / f"{safe_structure}_msd.h5")
        cond.dt.save_hdf5(self.data_path / f"{safe_structure}_dt.h5")
        np.save(
            self.data_path / f"{safe_structure}_distributions.npy",
            cond.distributions,
        )
