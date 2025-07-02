from pathlib import Path

import ase
import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx
import rdkit2ase
import znh5md
import zntrack
from rdkit import Chem
from scipy.signal import correlate

from massband.utils import unwrap_positions


class RadiusOfGyration(zntrack.Node):
    """Calculate the radius of gyration for molecules in a trajectory."""

    file: str | Path = zntrack.deps_path()
    suggestions: list[str] = zntrack.params(default_factory=list)
    figures: Path = zntrack.outs_path(zntrack.nwd / "figures")
    results: dict = zntrack.metrics()

    def get_data(self) -> dict:
        io = znh5md.IO(self.file, variable_shape=False, include=["position", "box"])
        frames: list[ase.Atoms] = io[:]
        positions = jnp.stack([atoms.positions for atoms in frames])
        cells = jnp.stack([atoms.cell[:] for atoms in frames])
        inv_cells = jnp.linalg.inv(cells)

        return {
            "positions": positions,
            "cells": cells,
            "inv_cells": inv_cells,
            "frames": frames,
        }

    def run(self) -> None:
        data = self.get_data()
        positions_unwrapped = unwrap_positions(
            data["positions"], data["cells"], data["inv_cells"]
        )

        atoms_0 = data["frames"][0]
        masses = jnp.array(atoms_0.get_masses())
        graph = rdkit2ase.ase2networkx(atoms_0, suggestions=self.suggestions)

        rg_by_smiles = {}
        self.results = {}

        for molecule_indices in nx.connected_components(graph):
            subgraph = graph.subgraph(molecule_indices)
            mol = rdkit2ase.networkx2rdkit(subgraph)
            mol = Chem.RemoveAllHs(mol)
            smiles = Chem.MolToSmiles(mol, canonical=True)

            rg = self._calculate_rg(molecule_indices, masses, positions_unwrapped)

            if smiles not in rg_by_smiles:
                rg_by_smiles[smiles] = []
            rg_by_smiles[smiles].append(rg)

        self.figures.mkdir(parents=True, exist_ok=True)

        all_rg_values = []

        for smiles, rg_values_list in rg_by_smiles.items():
            rg_values = jnp.array(rg_values_list)
            mean_rg = jnp.mean(rg_values, axis=0)
            std_rg = jnp.std(rg_values, axis=0)

            all_rg_values.append(rg_values)

            self.results[smiles] = {
                "mean": jnp.mean(mean_rg).item(),
                "std": jnp.mean(std_rg).item(),
            }

            # --- Time-series plot ---
            plt.figure()
            plt.plot(mean_rg, label="Mean RG")
            plt.fill_between(
                range(len(mean_rg)),
                mean_rg - std_rg,
                mean_rg + std_rg,
                alpha=0.2,
                label="Std Dev",
            )
            plt.xlabel("Frame")
            plt.ylabel("Radius of Gyration (Å)")
            plt.title(f"Radius of Gyration for {smiles}")
            plt.legend()
            filename_smiles = "".join(c for c in smiles if c.isalnum())
            plt.savefig(self.figures / f"rog_{filename_smiles}.png")
            plt.close()

            # --- Histogram ---
            plt.figure()
            plt.hist(rg_values.flatten(), bins=30, alpha=0.7)
            plt.xlabel("Radius of Gyration (Å)")
            plt.ylabel("Frequency")
            plt.title(f"Rg Distribution for {smiles}")
            plt.savefig(self.figures / f"hist_rog_{filename_smiles}.png")
            plt.close()

            # --- Autocorrelation ---
            ac = self._autocorrelation(mean_rg)
            plt.figure()
            plt.plot(ac)
            plt.xlabel("Lag")
            plt.ylabel("Autocorrelation")
            plt.title(f"Autocorrelation of Rg: {smiles}")
            plt.savefig(self.figures / f"ac_rog_{filename_smiles}.png")
            plt.close()

        # --- Global Rg analysis ---
        if all_rg_values:
            all_rg = jnp.concatenate(all_rg_values, axis=0)
            global_mean_rg = jnp.mean(all_rg, axis=0)
            global_std_rg = jnp.std(all_rg, axis=0)

            self.results["global"] = {
                "mean": jnp.mean(global_mean_rg).item(),
                "std": jnp.mean(global_std_rg).item(),
            }

            plt.figure()
            plt.plot(global_mean_rg, label="Global Mean RG")
            plt.fill_between(
                range(len(global_mean_rg)),
                global_mean_rg - global_std_rg,
                global_mean_rg + global_std_rg,
                alpha=0.2,
                label="Std Dev",
            )
            plt.xlabel("Frame")
            plt.ylabel("Radius of Gyration (Å)")
            plt.title("Global Mean Radius of Gyration")
            plt.legend()
            plt.savefig(self.figures / "global_rog.png")
            plt.close()

    def _calculate_rg(
        self,
        molecule_indices: jnp.ndarray,
        masses: jnp.ndarray,
        positions_unwrapped: jnp.ndarray,
    ) -> jnp.ndarray:
        molecule_indices = jnp.array(list(molecule_indices))
        molecule_masses = masses[molecule_indices]
        total_mass = jnp.sum(molecule_masses)
        molecule_positions = positions_unwrapped[:, molecule_indices, :]
        center_of_mass = (
            jnp.sum(molecule_masses[None, :, None] * molecule_positions, axis=1)
            / total_mass
        )
        displacements = molecule_positions - center_of_mass[:, None, :]
        squared_distances = jnp.sum(displacements**2, axis=2)
        mass_weighted_sq_dist = molecule_masses[None, :] * squared_distances
        sum_mass_weighted_sq_dist = jnp.sum(mass_weighted_sq_dist, axis=1)
        rg_squared = sum_mass_weighted_sq_dist / total_mass
        return jnp.sqrt(rg_squared)

    def _autocorrelation(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x - jnp.mean(x)
        result = correlate(x, x, mode="full")
        result = result[result.size // 2 :]
        result = result / result[0]
        return result
