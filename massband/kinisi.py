from dataclasses import dataclass

import numpy as np


@dataclass
class KinisiPlotData:
    """Dataclass to store data for plotting kinisi results."""

    structure: str
    dt: np.ndarray
    displacement: np.ndarray
    displacement_std: np.ndarray
    distribution: np.ndarray
    samples: np.ndarray
    mean_value: float
    start_dt: float
