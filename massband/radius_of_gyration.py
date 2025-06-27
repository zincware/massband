from functools import partial
from pathlib import Path

import ase
import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx
import rdkit2ase
import znh5md
import zntrack
from jax import vmap
from rdkit import Chem

from massband.utils import unwrap_positions


class RadiusOfGyration(zntrack.Node):
    """Calculate the radius of gyration for molecules in a trajectory.

    Attributes
    ----------
    file : str or Path
        The path to the trajectory file. This is a dependency for the node.
    suggestions : list[str], optional
        A list of SMILES strings to help with bond detection in `rdkit2ase`.
        Defaults to an empty list.
    figures : Path, optional
        The path to the directory where the output figures will be saved.
    """

    file: str | Path = zntrack.deps_path()
    suggestions: list[str] = zntrack.params(default_factory=list)
    figures: Path = zntrack.outs_path(zntrack.nwd / "figures")

    def get_data(self) -> dict:
        """Get data from the trajectory file.

        This method reads the trajectory file and returns a dictionary with
        positions, cells, inverse cells, and frames.

        Returns
        -------
        dict
            A dictionary with the following keys:
            - positions: The positions of the atoms in the trajectory.
            - cells: The cell vectors of the trajectory.
            - inv_cells: The inverse cell vectors of the trajectory.
            - frames: The list of ase.Atoms objects.
        """
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
        """Run the radius of gyration calculation.

        This method calculates the radius of gyration for each molecule in the
        trajectory and saves the results as plots.
        """
        data = self.get_data()
        unwrap_positions_vmap = vmap(
            partial(unwrap_positions, cells=data["cells"], inv_cells=data["inv_cells"]),
            in_axes=(1,),
        )
        # positions_unwrapped will have shape (n_atoms, n_frames, 3)
        positions_unwrapped = unwrap_positions_vmap(data["positions"])

        atoms_0 = data["frames"][0]
        masses = jnp.array(atoms_0.get_masses())
        graph = rdkit2ase.ase2networkx(atoms_0, suggestions=self.suggestions)

        rg_by_smiles = {}

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

        for smiles, rg_values_list in rg_by_smiles.items():
            rg_values = jnp.array(rg_values_list)
            mean_rg = jnp.mean(rg_values, axis=0)
            std_rg = jnp.std(rg_values, axis=0)

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
            # Sanitize smiles for filename
            filename_smiles = "".join(c for c in smiles if c.isalnum())
            plt.savefig(self.figures / f"rog_{filename_smiles}.png")
            plt.close()

    def _calculate_rg(
        self,
        molecule_indices: jnp.ndarray,
        masses: jnp.ndarray,
        positions_unwrapped: jnp.ndarray,
    ) -> jnp.ndarray:
        """Calculate the radius of gyration for a single molecule.

        Parameters
        ----------
        molecule_indices : jnp.ndarray
            The indices of the atoms in the molecule.
        masses : jnp.ndarray
            The masses of the atoms.
        positions_unwrapped : jnp.ndarray
            The unwrapped positions of the atoms.

        Returns
        -------
        jnp.ndarray
            The radius of gyration for each frame.
        """
        molecule_indices = jnp.array(list(molecule_indices))
        molecule_masses = masses[molecule_indices]
        total_mass = jnp.sum(molecule_masses)
        molecule_positions = positions_unwrapped[molecule_indices, :, :]
        center_of_mass = (
            jnp.sum(molecule_masses[:, None, None] * molecule_positions, axis=0)
            / total_mass
        )
        displacements = molecule_positions - center_of_mass[None, :, :]
        squared_distances = jnp.sum(displacements**2, axis=2)
        mass_weighted_sq_dist = molecule_masses[:, None] * squared_distances
        sum_mass_weighted_sq_dist = jnp.sum(mass_weighted_sq_dist, axis=0)
        rg_squared = sum_mass_weighted_sq_dist / total_mass
        return jnp.sqrt(rg_squared)
