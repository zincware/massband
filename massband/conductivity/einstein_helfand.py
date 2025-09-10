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
from kinisi.analyze import ConductivityAnalyzer
from rdkit import Chem

log = logging.getLogger(__name__)



class KinisiEinsteinHelfandIonicConductivity(zntrack.Node):
    """Compute ionic conductivity using the kinisi library.

    Analyzes molecular dynamics trajectories to calculate system-wide ionic conductivity
    and mean squared charge displacements for all ionic species combined.

    Parameters
    ----------
    file : Union[str, Path]
        Path to the trajectory file in h5md format.
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
    conductivity : dict[str, float]
        Dictionary containing system-wide conductivity coefficient and standard deviation.
        Keys: 'mean', 'std', 'var'.

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
    dict_keys(['mean', 'std', 'var'])

    References
    ----------
    .. [1] https://kinisi.readthedocs.io/en/stable/
    """

    file: Union[str, Path] = zntrack.deps_path()
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

    conductivity: dict[str, float] = zntrack.metrics()

    def _build_charge_mapping(self) -> dict[str, int]:
        """Build charge mapping from SMILES structures.

        Returns
        -------
        dict[str, int]
            Dictionary mapping structure SMILES to their formal charges.

        Raises
        ------
        ValueError
            If any SMILES string is invalid.
        """
        charge_mapping = {}
        for structure in self.structures:
            mol = Chem.MolFromSmiles(structure)
            if mol is not None:
                charge = Chem.GetFormalCharge(mol)
                if charge != 0:
                    charge_mapping[structure] = charge
            else:
                raise ValueError(f"Invalid SMILES string: {structure}")

        print("Charge mapping:", charge_mapping)
        return charge_mapping


    def run(self):
        self.data_path.mkdir(exist_ok=True, parents=True)
        self.figures_path.mkdir(exist_ok=True, parents=True)