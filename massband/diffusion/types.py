"""Type definitions for diffusion analysis."""

import typing as t


class DiffusionData(t.TypedDict):
    """Data structure for diffusion coefficient analysis results.

    Attributes
    ----------
    mean : float
        Mean diffusion coefficient value.
    std : float
        Standard deviation of diffusion coefficient.
    var : float
        Variance of diffusion coefficient.
    occurrences : int
        Number of molecules/ions of this type in the system.
    unit : str
        Unit of the diffusion coefficient (e.g., "cm^2/s").
    box : list[list[float]] | None
        Simulation box cell array (3x3 matrix) in angstroms, or None if not available.
    """

    mean: float
    std: float
    var: float
    occurrences: int | None
    unit: str
    box: list[list[float]] | None
