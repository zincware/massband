import logging
import typing as t
from pathlib import Path

import ase
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pint
import rdkit2ase
import znh5md
import zntrack
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from scipy.signal import correlate

from massband.utils import sanitize_structure_name

log = logging.getLogger(__name__)


class ROGData(t.TypedDict):
    """A data structure for storing Radius of Gyration statistics."""

    mean: float
    std: float
    mean_ensemble: float | None
    std_ensemble: float | None
    ensemble: list[float] | None
    unit: str


def _apply_cea_reweighting_rog(
    rog_per_frame: np.ndarray,
    energy_model: np.ndarray,
    energy_mean: np.ndarray,
    temperature: float,
) -> float:
    """
    Apply Cumulant Expansion Approximation (CEA) reweighting to ROG time series.

    The CEA formula: O_CEA = <O> - <O*h> + <O>*<h>
    where h = beta * (E_model - E_mean)

    This method propagates ensemble energy uncertainties to ROG uncertainties
    and is related to umbrella sampling reweighting methods.

    Parameters
    ----------
    rog_per_frame : np.ndarray
        ROG value per frame, shape (n_frames,)
    energy_model : np.ndarray
        Energy for specific model, shape (n_frames,)
    energy_mean : np.ndarray
        Mean energy across ensemble, shape (n_frames,)
    temperature : float
        Temperature in Kelvin

    Returns
    -------
    rog_cea : float
        CEA-reweighted mean ROG value

    References
    ----------
    Torrie, Glenn M., and John P. Valleau. "Nonphysical sampling distributions
    in Monte Carlo free-energy estimation: Umbrella sampling."
    Journal of Computational Physics 23.2 (1977): 187-199.
    https://doi.org/10.1016/0021-9991(77)90121-8
    """
    # Compute beta = 1/(k_B * T) in eV^-1
    ureg = pint.UnitRegistry()
    k_B = ureg.boltzmann_constant
    beta = (1.0 / (k_B * temperature * ureg.kelvin)).to("1/eV").magnitude

    # Compute CEA reweighting factor
    cea_factor = beta * (energy_model - energy_mean)  # (n_frames,)

    # Apply CEA formula
    O_mean = np.mean(rog_per_frame)
    Oh_mean = np.mean(rog_per_frame * cea_factor)
    h_mean = np.mean(cea_factor)

    rog_cea = O_mean - Oh_mean + O_mean * h_mean

    return float(rog_cea)


class RadiusOfGyration(zntrack.Node):
    """Calculate the radius of gyration for molecules in a trajectory.

    Parameters
    ----------
    file : str | Path | None
        Path to trajectory file in H5MD format.
    data: znh5md.IO | list[ase.Atoms] | None, default None
        znh5md.IO object for trajectory data, as an alternative to 'file'.
    structures : list[str]
        SMILES strings for molecular structures to analyze.
    uncertainties : str or None, default None
        Name of the property containing ensemble energies (e.g., "energy_ensemble").
        If provided, computes uncertainty-aware ROG using the Cumulant Expansion
        Approximation (CEA) method. Requires temperature parameter.
    temperature : float or None, default None
        Temperature in Kelvin. Required when uncertainties is set.
    """

    data: znh5md.IO | list[ase.Atoms] = zntrack.deps()
    structures: list[str] = zntrack.params(default_factory=list)
    figures_path: Path = zntrack.outs_path(zntrack.nwd / "figures")
    data_path: Path = zntrack.outs_path(zntrack.nwd / "data")
    rog: dict[str, ROGData] = zntrack.metrics()

    start: int = zntrack.params(0)
    stop: int | None = zntrack.params(None)
    step: int = zntrack.params(1)
    uncertainties: str | None = zntrack.params(None)
    temperature: float | None = zntrack.params(None)

    sampling_rate: float | None = zntrack.params(
        None
    )  # Optional time between frames in fs
    time_step: float | None = zntrack.params(None)  # Time step in fs

    def run(self) -> None:
        """Main method to execute the ROG calculation."""
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.figures_path.mkdir(parents=True, exist_ok=True)
        self.rog = {}

        # --- Uncertainty Setup ---
        use_uncertainty = self.uncertainties is not None
        if use_uncertainty:
            if self.temperature is None:
                raise ValueError(
                    "temperature parameter must be set when uncertainties is enabled"
                )
            log.info(
                f"Uncertainty quantification enabled using '{self.uncertainties}' "
                f"at T={self.temperature} K (CEA method)"
            )

        # --- Data Loading ---

        io = self.data
        if isinstance(io, znh5md.IO):
            include_list = ["position", "box"]
            if use_uncertainty:
                include_list.append(self.uncertainties)
            io.include = include_list
        frames = io[self.start : self.stop : self.step]

        # --- Extract Energy Ensemble Data ---
        energy_ensemble = None
        if use_uncertainty:
            log.info("Extracting energy ensemble data from frames...")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
            ) as progress:
                energy_task = progress.add_task(
                    "[cyan]Extracting energies...", total=len(frames)
                )
                energy_list = []
                for frame in frames:
                    if hasattr(frame, "calc") and frame.calc is not None:
                        ensemble = frame.calc.results.get(self.uncertainties, None)
                        if ensemble is None:
                            raise ValueError(
                                f"Frame missing '{self.uncertainties}' in calc.results"
                            )
                        energy_list.append(ensemble)
                    else:
                        raise ValueError(
                            f"Frame has no calculator attached. Cannot extract '{self.uncertainties}'"
                        )
                    progress.update(energy_task, advance=1)
            energy_ensemble = np.array(energy_list)
            log.info(f"Extracted energy ensemble: shape {energy_ensemble.shape}")

        # --- Molecule Identification ---
        graph = rdkit2ase.ase2networkx(frames[0], suggestions=self.structures)
        molecules: dict[str, tuple[tuple[int, ...]]] = {}
        masses: dict[str, np.ndarray] = {}

        if not self.structures:
            raise ValueError("No structures provided for ROG calculation.")
        else:
            for structure in self.structures:
                matches = rdkit2ase.match_substructure(
                    rdkit2ase.networkx2ase(graph), smiles=structure
                )
                if not matches:
                    log.warning(f"No matches found for structure {structure}")
                    continue
                molecules[structure] = matches
                masses[structure] = np.array(frames[0].get_masses()[list(matches[0])])

        # --- ROG Calculation ---
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            overall_task = progress.add_task(
                "[bold green]Overall Progress", total=len(molecules)
            )

            for structure, mol_indices in molecules.items():
                safe_structure = sanitize_structure_name(structure)
                molecule_masses = masses[structure]
                total_mass = np.sum(molecule_masses)
                rog_time_series = []

                # Add progress bar for processing frames
                frame_task = progress.add_task(
                    f"[cyan]Processing {structure}", total=len(frames)
                )
                for frame in frames:
                    positions = np.array(frame.get_positions())
                    box_lengths = np.diag(np.array(frame.get_cell()))

                    rog_per_frame = []
                    for idx_tuple in mol_indices:
                        mol_idx = np.array(idx_tuple)
                        mol_coords = positions[mol_idx]

                        # This assumes an orthorhombic box.
                        ref_coord = mol_coords[0]
                        displacements = mol_coords - ref_coord
                        displacements -= box_lengths * np.round(
                            displacements / box_lengths
                        )
                        unwrapped_coords = ref_coord + displacements

                        # Calculate Center of Mass (COM)
                        com = (
                            np.sum(unwrapped_coords * molecule_masses[:, None], axis=0)
                            / total_mass
                        )

                        # Calculate Radius of Gyration
                        r_minus_com = unwrapped_coords - com
                        rog_sq = (
                            np.sum(molecule_masses * np.sum(r_minus_com**2, axis=1))
                            / total_mass
                        )
                        rog_per_frame.append(np.sqrt(rog_sq))

                    rog_time_series.append(np.mean(np.array(rog_per_frame)))
                    progress.update(frame_task, advance=1)

                # Remove the frame task to clean up before next structure
                progress.remove_task(frame_task)

                # --- Data Processing and Saving ---
                rog_time_series = np.array(rog_time_series)

                # Compute ensemble statistics if uncertainties are enabled
                mean_rog_ensemble = None
                std_rog_ensemble = None
                rog_ensemble_values = None

                if use_uncertainty:
                    assert energy_ensemble is not None
                    assert self.temperature is not None

                    # Compute mean energy per frame across ensemble
                    energy_mean = np.mean(energy_ensemble, axis=1)  # (n_frames,)

                    # Apply CEA for each model
                    n_models = energy_ensemble.shape[1]
                    rog_per_model = []
                    log.info(
                        f"Applying CEA reweighting for {n_models} models in ensemble..."
                    )
                    for model_idx in range(n_models):
                        energy_model = energy_ensemble[:, model_idx]  # (n_frames,)

                        # Apply CEA reweighting
                        rog_cea = _apply_cea_reweighting_rog(
                            rog_time_series, energy_model, energy_mean, self.temperature
                        )
                        rog_per_model.append(rog_cea)

                    # Compute ensemble statistics
                    rog_ensemble_values = np.array(rog_per_model)
                    mean_rog_ensemble = np.mean(rog_ensemble_values)
                    std_rog_ensemble = np.std(rog_ensemble_values)

                    log.info(
                        f"Structure: {structure} | CEA Mean ROG: "
                        f"{mean_rog_ensemble:.3f} ± {std_rog_ensemble:.3f} Angstrom"
                    )

                # Save time series data
                df = pd.DataFrame(
                    {
                        "Time": np.arange(len(rog_time_series))
                        * (self.sampling_rate if self.sampling_rate else 1),
                        "ROG": rog_time_series,
                    }
                )
                df.to_csv(self.data_path / f"{safe_structure}_rog.csv", index=False)

                # Save ensemble data if available
                if use_uncertainty and rog_ensemble_values is not None:
                    ensemble_df = pd.DataFrame(
                        {
                            "Model": np.arange(len(rog_ensemble_values)),
                            "ROG": rog_ensemble_values,
                        }
                    )
                    ensemble_df.to_csv(
                        self.data_path / f"{safe_structure}_rog_ensemble.csv", index=False
                    )

                # Compute time-series statistics (backward compatibility)
                mean_rog = np.mean(rog_time_series)
                std_rog = np.std(rog_time_series)

                # Save metrics dictionary
                self.rog[structure] = {
                    "mean": float(mean_rog),
                    "std": float(std_rog),
                    "mean_ensemble": (
                        float(mean_rog_ensemble)
                        if mean_rog_ensemble is not None
                        else None
                    ),
                    "std_ensemble": (
                        float(std_rog_ensemble) if std_rog_ensemble is not None else None
                    ),
                    "ensemble": (
                        rog_ensemble_values.tolist()
                        if rog_ensemble_values is not None
                        else None
                    ),
                    "unit": "Angstrom",
                }
                log.info(
                    f"Structure: {structure} | Time-series Mean ROG: {mean_rog:.3f} ± {std_rog:.3f} Angstrom"
                )

                # --- Plotting ---
                # 1. ROG over time
                plt.figure(figsize=(10, 6))
                time_np = np.arange(len(rog_time_series)) * self.step
                if self.sampling_rate and self.time_step:
                    time_step = self.time_step * pint.Quantity(self.sampling_rate, "fs")
                    time_np = time_np * time_step.magnitude
                plt.plot(time_np, rog_time_series, label="ROG (time series)", alpha=0.7)

                # Add ensemble uncertainty as horizontal band if available
                if (
                    use_uncertainty
                    and mean_rog_ensemble is not None
                    and std_rog_ensemble is not None
                ):
                    # Plot 95% confidence interval (±1.96 * σ)
                    ci_factor = 1.96
                    plt.axhline(
                        mean_rog_ensemble,
                        color="red",
                        linestyle="--",
                        linewidth=2,
                        label="CEA Mean",
                    )
                    plt.axhspan(
                        mean_rog_ensemble - ci_factor * std_rog_ensemble,
                        mean_rog_ensemble + ci_factor * std_rog_ensemble,
                        alpha=0.2,
                        color="red",
                        label="95% CI (CEA)",
                    )

                plt.xlabel(f"Lag Time / {'fs' if self.sampling_rate else 'Frames'}")
                plt.ylabel("Radius of Gyration / Angstrom")
                title = f"ROG of {structure} over Time"
                if use_uncertainty:
                    title += f" (T={self.temperature} K)"
                plt.title(title)
                plt.legend()
                plt.grid(True, linestyle="--", alpha=0.6)
                plt.tight_layout()
                plt.savefig(
                    self.figures_path / f"{safe_structure}_rog_vs_time.png", dpi=300
                )
                plt.close()

                # 2. ROG distribution
                plt.figure(figsize=(8, 6))
                plt.hist(
                    rog_time_series,
                    bins=50,
                    density=True,
                    alpha=0.75,
                    label="Time series",
                )

                # Add ensemble uncertainty indicators if available
                if (
                    use_uncertainty
                    and mean_rog_ensemble is not None
                    and std_rog_ensemble is not None
                ):
                    plt.axvline(
                        mean_rog_ensemble,
                        color="red",
                        linestyle="--",
                        linewidth=2,
                        label=f"CEA Mean: {mean_rog_ensemble:.3f} Å",
                    )
                    ci_factor = 1.96
                    plt.axvline(
                        mean_rog_ensemble - ci_factor * std_rog_ensemble,
                        color="red",
                        linestyle=":",
                        linewidth=1.5,
                        alpha=0.7,
                        label="95% CI",
                    )
                    plt.axvline(
                        mean_rog_ensemble + ci_factor * std_rog_ensemble,
                        color="red",
                        linestyle=":",
                        linewidth=1.5,
                        alpha=0.7,
                    )

                plt.xlabel("Radius of Gyration / Angstrom")
                plt.ylabel("Probability Density")
                title = f"ROG Distribution for {structure}"
                if use_uncertainty:
                    title += f" (T={self.temperature} K)"
                plt.title(title)
                plt.legend()
                plt.grid(True, linestyle="--", alpha=0.6)
                plt.tight_layout()
                plt.savefig(
                    self.figures_path / f"{safe_structure}_rog_distribution.png", dpi=300
                )
                plt.close()

                # 3. ROG autocorrelation function
                rog_centered = rog_time_series - mean_rog
                acf = correlate(rog_centered, rog_centered, mode="full", method="fft")
                acf = acf[len(acf) // 2 :]
                acf /= acf[0]

                # Plot up to a reasonable lag time
                max_lag_idx = min(
                    len(acf),
                    np.where(acf < 1 / np.e)[0][0] * 3
                    if np.any(acf < 1 / np.e)
                    else len(acf),
                )

                plt.figure(figsize=(10, 6))
                plt.plot(time_np[:max_lag_idx], acf[:max_lag_idx])
                plt.xlabel(f"Lag Time / {'fs' if self.sampling_rate else 'Frames'}")
                plt.ylabel("Autocorrelation")
                plt.title(f"ROG Autocorrelation for {structure}")
                plt.axhline(0, color="gray", linestyle="--")
                plt.grid(True, linestyle="--", alpha=0.6)
                plt.tight_layout()
                plt.savefig(self.figures_path / f"{safe_structure}_rog_acf.png", dpi=300)
                plt.close()

                # 4. Ensemble spread plot (if uncertainties enabled)
                if use_uncertainty and rog_ensemble_values is not None:
                    plt.figure(figsize=(10, 6))
                    model_indices = np.arange(len(rog_ensemble_values))
                    plt.bar(
                        model_indices,
                        rog_ensemble_values,
                        alpha=0.7,
                        color="steelblue",
                        edgecolor="black",
                    )
                    plt.axhline(
                        mean_rog_ensemble,
                        color="red",
                        linestyle="--",
                        linewidth=2,
                        label=f"Mean: {mean_rog_ensemble:.3f} Å",
                    )
                    plt.axhline(
                        mean_rog_ensemble + 1.96 * std_rog_ensemble,
                        color="red",
                        linestyle=":",
                        linewidth=1.5,
                        alpha=0.7,
                        label="95% CI",
                    )
                    plt.axhline(
                        mean_rog_ensemble - 1.96 * std_rog_ensemble,
                        color="red",
                        linestyle=":",
                        linewidth=1.5,
                        alpha=0.7,
                    )
                    plt.xlabel("Model Index")
                    plt.ylabel("Radius of Gyration / Angstrom")
                    plt.title(
                        f"Ensemble ROG Spread for {structure} (CEA, T={self.temperature} K)"
                    )
                    plt.legend()
                    plt.grid(True, linestyle="--", alpha=0.6, axis="y")
                    plt.tight_layout()
                    plt.savefig(
                        self.figures_path / f"{safe_structure}_rog_ensemble.png", dpi=300
                    )
                    plt.close()

                # Update overall progress after completing this structure
                progress.update(overall_task, advance=1)
