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
from scipy.signal import correlate
from tqdm import tqdm

log = logging.getLogger(__name__)


class ROGData(t.TypedDict):
    """A data structure for storing Radius of Gyration statistics."""

    mean: float
    std: float
    unit: str


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
    """

    file: str | Path | None = zntrack.deps_path()
    data: znh5md.IO | list[ase.Atoms] | None = zntrack.deps(None)
    structures: list[str] = zntrack.params(default_factory=list)
    figures_path: Path = zntrack.outs_path(zntrack.nwd / "figures")
    data_path: Path = zntrack.outs_path(zntrack.nwd / "data")
    rog: dict[str, ROGData] = zntrack.metrics()

    start: int = zntrack.params(0)
    stop: int | None = zntrack.params(None)
    step: int = zntrack.params(1)

    sampling_rate: float | None = zntrack.params(
        None
    )  # Optional time between frames in fs
    time_step: float | None = zntrack.params(None)  # Time step in fs

    def run(self) -> None:
        """Main method to execute the ROG calculation."""
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.figures_path.mkdir(parents=True, exist_ok=True)
        self.rog = {}

        # --- Data Loading ---
        if self.data is not None and self.file is not None:
            raise ValueError("Provide either 'data' or 'file', not both.")
        elif self.file is not None:
            io = znh5md.IO(self.file, include=["position", "box"])
        elif self.data is not None:
            io = self.data
            if isinstance(io, znh5md.IO):
                io.include = ["position", "box"]
        else:
            raise ValueError("Either 'file' or 'data' must be provided.")
        frames = io[self.start : self.stop : self.step]
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
        for structure, mol_indices in molecules.items():
            molecule_masses = masses[structure]
            total_mass = np.sum(molecule_masses)
            rog_time_series = []

            # Add progress bar for processing frames
            pbar = tqdm(frames, desc=f"Processing {structure}")
            for frame in pbar:
                positions = np.array(frame.get_positions())
                box_lengths = np.diag(np.array(frame.get_cell()))

                rog_per_frame = []
                for idx_tuple in mol_indices:
                    mol_idx = np.array(idx_tuple)
                    mol_coords = positions[mol_idx]

                    # This assumes an orthorhombic box.
                    ref_coord = mol_coords[0]
                    displacements = mol_coords - ref_coord
                    displacements -= box_lengths * np.round(displacements / box_lengths)
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

            # --- Data Processing and Saving ---
            rog_time_series = np.array(rog_time_series)
            df = pd.DataFrame(
                {
                    "Time": np.arange(len(rog_time_series))
                    * (self.sampling_rate if self.sampling_rate else 1),
                    "ROG": rog_time_series,
                }
            )
            df.to_csv(self.data_path / f"{structure}_rog.csv", index=False)

            # Save mean and std to metrics dictionary
            mean_rog = np.mean(rog_time_series)
            std_rog = np.std(rog_time_series)
            self.rog[structure] = {
                "mean": float(mean_rog),
                "std": float(std_rog),
                "unit": "Angstrom",
            }
            log.info(
                f"Structure: {structure} | Mean ROG: {mean_rog:.3f} ± {std_rog:.3f} Angstrom"
            )

            # --- Plotting ---
            # 1. ROG over time
            plt.figure(figsize=(10, 6))
            time_np = np.arange(len(rog_time_series)) * self.step
            if self.sampling_rate and self.time_step:
                time_step = self.time_step * pint.Quantity(self.sampling_rate, "fs")
                time_np = time_np * time_step.magnitude
            plt.plot(time_np, rog_time_series, label="ROG")
            plt.xlabel(f"Lag Time / {'fs' if self.sampling_rate else 'Frames'}")
            plt.ylabel("Radius of Gyration / Angstrom")
            plt.title(f"ROG of {structure} over Time")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig(self.figures_path / f"{structure}_rog_vs_time.png", dpi=300)
            plt.close()

            # 2. ROG distribution
            plt.figure(figsize=(8, 6))
            plt.hist(rog_time_series, bins=50, density=True, alpha=0.75)
            plt.xlabel("Radius of Gyration / Angstrom")
            plt.ylabel("Probability Density")
            plt.title(f"ROG Distribution for {structure}")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig(self.figures_path / f"{structure}_rog_distribution.png", dpi=300)
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
            plt.savefig(self.figures_path / f"{structure}_rog_acf.png", dpi=300)
            plt.close()
