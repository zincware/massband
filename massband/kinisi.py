from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass
class KinisiPlotData:
    """Dataclass to store data for plotting kinisi results.

    This class contains all the necessary data for generating kinisi plots,
    including displacement data, samples, and statistical measures.

    Attributes
    ----------
    structure : str
        Name/identifier of the molecular structure.
    dt : npt.NDArray[np.floating[Any]]
        Array of time deltas (time intervals) for the analysis.
    displacement : npt.NDArray[np.floating[Any]]
        Array of displacement values (e.g., MSD or MSCD).
    displacement_std : npt.NDArray[np.floating[Any]]
        Array of standard deviations for displacement values.
    distribution : npt.NDArray[np.floating[Any]]
        Array containing the distribution of samples over time.
    samples : npt.NDArray[np.floating[Any]]
        Array of bootstrap samples used for statistical analysis.
    mean_value : float
        Mean value of the measured property (e.g., diffusion coefficient).
    start_dt : float
        Starting time delta for the analysis in picoseconds.
    """

    structure: str
    dt: npt.NDArray[np.floating[Any]]
    displacement: npt.NDArray[np.floating[Any]]
    displacement_std: npt.NDArray[np.floating[Any]]
    distribution: npt.NDArray[np.floating[Any]]
    samples: npt.NDArray[np.floating[Any]]
    mean_value: float
    start_dt: float
