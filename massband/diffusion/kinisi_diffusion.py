import zntrack

import jax.numpy as jnp
import znh5md
import ase
import logging
from massband.utils import unwrap_positions
from jax import vmap
from functools import partial
from collections import defaultdict
import matplotlib.pyplot as plt
import rdkit2ase
import numpy as np

from kinisi.analyze import DiffusionAnalyzer


log = logging.getLogger(__name__)


class KinisiSelfDiffusion(zntrack.Node):
    file: str = zntrack.deps_path()
    sampling_rate: int = zntrack.params()
    time_step: float = zntrack.params()
    structures: list[str]|None = zntrack.params(None)
    start_dt: float = zntrack.params(50) #  in ps
    
    def get_data(self):
        io = znh5md.IO(
            self.file, variable_shape=False, include=["position", "box"],
        )
        # frames = io[:]
        frames = io[:] # testing!
        positions = jnp.stack([atoms.positions for atoms in frames])
        cells = jnp.stack([atoms.cell[:] for atoms in frames])
        inv_cells = jnp.linalg.inv(cells)
        atomic_numbers = frames[0].get_atomic_numbers().tolist()
        masses = frames[0].get_masses().tolist()

        return {
            "positions": positions,
            "cells": cells,
            "inv_cells": inv_cells,
            "atomic_numbers": atomic_numbers,
            "masses": masses,
            "frames": frames
        }

    def run(self):
        data = self.get_data()
        positions = data["positions"]
        cells = data["cells"]
        inv_cells = data["inv_cells"]
        masses = data["masses"]
        frames = data["frames"]
        # unwrap positions, unwrap_positions takes (n_frames, 3) so we need to map the position vectors
        unwrap_positions_vmap = vmap(
            partial(unwrap_positions, cells=cells, inv_cells=inv_cells), in_axes=(1,)
        )
        positions_unwrapped = unwrap_positions_vmap(positions)
        print(f"Unwrapped positions shape: {positions_unwrapped.shape}") # (n_atoms, n_frames, 3)
        
        substructures = defaultdict(list)
        if self.structures is not None:
            log.info(f"Searching for substructures in {len(self.structures)} patterns")
            for structure in self.structures:
                indices = rdkit2ase.match_substructure(
                    frames[0], smiles=structure, suggestions=self.structures,
                )
                if len(indices) > 0:
                    substructures[structure].extend(indices)

                log.info(f"Found {len(indices)} matches for substructure {structure} in the dataset.")

        # for each type of molecule, create a custom ase atoms object
        for structure, indices in substructures.items():
            log.info(f"Creating ASE Atoms object for substructure {structure} with indices {indices}")
            # flatten the indices list
            flatt_indices = [item for sublist in indices for item in sublist]
            sub_frames = [atoms[flatt_indices] for atoms in frames]
            # need to recompute / rematch to get the correct indices
            # TODO: rematching here was quicker to implement but is slower!
            specie_indices = rdkit2ase.match_substructure(
                    sub_frames[0], smiles=structure, suggestions=self.structures,
                )
            masses = sub_frames[0][specie_indices[0]].get_masses().tolist()

            p_parms = {'specie': None,
                'time_step': self.time_step / 1000,  # convert fs to ps
                'step_skip': self.sampling_rate,
                'specie_indices': specie_indices,
                'masses': masses,
                'progress': False
                }

            u_params = {'progress': False}

            diff = DiffusionAnalyzer.from_ase(sub_frames, parser_params=p_parms, uncertainty_params=u_params)

            plt.errorbar(diff.dt, diff.msd, diff.msd_std)
            plt.ylabel('MSD/Å$^2$')
            plt.xlabel(r'$\Delta t$/ps')
            plt.show()

            diff.diffusion(self.start_dt, {'progress': False})

            credible_intervals = [[16, 84], [2.5, 97.5], [0.15, 99.85]]
            alpha = [0.6, 0.4, 0.2]

            plt.plot(diff.dt, diff.msd, 'k-')
            for i, ci in enumerate(credible_intervals):
                plt.fill_between(diff.dt,
                                *np.percentile(diff.distribution, ci, axis=1),
                                alpha=alpha[i],
                                color='#0173B2',
                                lw=0)
            plt.ylabel('MSD/Å$^2$')
            plt.xlabel(r'$\Delta t$/ps')
            # TODO: plot a horizontal line where the diffusion computataion starts, e.g. self.start_dt
            plt.axvline(self.start_dt, c='k', ls='--')
            plt.show()


            plt.hist(diff.D.samples, density=True)
            plt.axvline(diff.D.n, c='k')
            plt.xlabel('$D$/cm$^2$s$^{-1}$')
            plt.ylabel('$p(D$/cm$^2$s$^{-1})$')
            plt.show()