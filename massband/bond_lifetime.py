import typing as tp
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import zntrack
from jax.scipy.signal import fftconvolve
from rdkit import Chem
from tqdm import tqdm

from massband.dataloader import TimeBatchedLoader
from massband.rdf.utils import select_atoms_flat_unique, visualize_selected_molecules


def _get_molecule_ids(mol: Chem.Mol) -> np.ndarray:
    """Creates an array mapping each atom index to a molecule ID."""
    mol_ids = np.zeros(mol.GetNumAtoms(), dtype=int)
    for mol_id, atom_indices in enumerate(Chem.GetMolFrags(mol, asMols=False)):
        mol_ids[list(atom_indices)] = mol_id
    return mol_ids


def _compute_vectors_pbc(pos_a, pos_b, cell):
    """Compute displacement vectors under periodic boundary conditions."""
    delta = pos_a - pos_b
    inv_cell = jnp.linalg.inv(cell)
    scaled_delta = delta @ inv_cell
    scaled_delta -= jnp.round(scaled_delta)
    return scaled_delta @ cell


@jax.jit
def _compute_bond_matrix(
    positions: jnp.ndarray,
    cell: jnp.ndarray,
    indices_a: jnp.ndarray,
    indices_b: jnp.ndarray,
    mol_ids_a: jnp.ndarray,
    mol_ids_b: jnp.ndarray,
    distance_cutoff: float,
    exclude_self: bool,
) -> jnp.ndarray:
    """Computes a boolean matrix indicating bond formation based on distance."""
    pos_a, pos_b = positions[indices_a], positions[indices_b]
    pos_a_exp, pos_b_exp = pos_a[:, jnp.newaxis, :], pos_b[jnp.newaxis, :, :]

    vec_ab = _compute_vectors_pbc(pos_a_exp, pos_b_exp, cell)
    dist_ab = jnp.linalg.norm(vec_ab, axis=-1)

    intramolecular_mask = mol_ids_a[:, jnp.newaxis] == mol_ids_b[jnp.newaxis, :]
    dist_ab = jax.lax.cond(
        exclude_self,
        lambda d: jnp.where(intramolecular_mask, jnp.inf, d),
        lambda d: d,
        dist_ab,
    )
    return dist_ab < distance_cutoff


T_H_OPT = tp.Literal["include", "exclude", "isolated"]


class SubstructureBondLifetime(zntrack.Node):
    """Calculate bond lifetimes for selected substructure pairs."""

    # --- Inputs ---
    file: Path = zntrack.deps_path()
    structures: list[str] = zntrack.params(default_factory=list)
    pairs: list[tuple[str, str]] = zntrack.params(default_factory=list)
    hydrogens: list[tuple[T_H_OPT, T_H_OPT]] = zntrack.params(default_factory=list)
    distance_cutoff: float = zntrack.params(3.5)
    batch_size: int = zntrack.params(64)
    start: int = zntrack.params(0)
    stop: int | None = zntrack.params(None)
    step: int = zntrack.params(1)
    timestep: float = zntrack.params(0.5)
    sampling_rate: int = zntrack.params(2000)

    # --- Outputs ---
    figures: Path = zntrack.outs_path(zntrack.nwd / "figures")
    results: dict = zntrack.outs()

    def _plot_results(self, lifetime_data):
        """Create and save plots for each pair's autocorrelation function."""
        self.figures.mkdir(parents=True, exist_ok=True)

        for pair_idx, data in lifetime_data.items():
            pair_smarts = self.pairs[pair_idx]
            time_ps = np.array(data["time_axis_ps"])
            autocorr = np.array(data["autocorrelation"])
            lifetime_int_ps = data["lifetime_integrated_ps"]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(
                f"Bond Lifetime: {pair_smarts[0]} – {pair_smarts[1]}\n(Distance Cutoff = {self.distance_cutoff} Å)",
                fontsize=14,
            )

            ax1.plot(
                time_ps,
                autocorr,
                "o",
                markersize=4,
                label=f"Simulation Data\nτ (integrated) = {lifetime_int_ps:.2f} ps",
            )
            ax1.set_xlabel("Time / ps")
            ax1.set_ylabel("Bond Autocorrelation C(t)")
            ax1.set_title("Linear Scale")
            ax1.grid(True, linestyle="--", alpha=0.6)

            ax2.plot(time_ps, autocorr, "o", markersize=4)
            ax2.set_xlabel("Time / ps")
            ax2.set_ylabel("Bond Autocorrelation C(t) (log scale)")
            ax2.set_title("Log Scale")
            ax2.grid(True, linestyle="--", alpha=0.6)
            ax2.set_yscale("log")

            ax1.legend()
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            safe_smarts1 = (
                pair_smarts[0].replace("[", "").replace("]", "").replace(":", "_")
            )
            safe_smarts2 = (
                pair_smarts[1].replace("[", "").replace("]", "").replace(":", "_")
            )
            plot_path = (
                self.figures
                / f"lifetime_pair_{pair_idx}_{safe_smarts1}_{safe_smarts2}.png"
            )
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"Saved lifetime plot to {plot_path}")

    def run(self):
        """Main execution method for the node."""
        self.figures.mkdir(parents=True, exist_ok=True)
        dl = TimeBatchedLoader(
            file=self.file,
            batch_size=self.batch_size,
            structures=self.structures,
            wrap=False,
            properties=["position", "cell"],
            start=self.start,
            stop=self.stop,
            step=self.step,
            com=False,
            map_to_dict=False,
        )
        # Correctly calculate the effective time between analyzed frames in femtoseconds
        time_step_fs = self.timestep * self.sampling_rate * self.step

        ## --------------------------------------------------------------------
        ## Step 1: Select atoms and prepare molecule IDs
        ## --------------------------------------------------------------------
        print("Step 1: Selecting atoms and preparing molecule IDs...")
        molecule_ids = _get_molecule_ids(dl.first_frame_chem)

        all_pair_data = []
        for pair_idx, ((smarts1, smarts2), (h1, h2)) in enumerate(
            zip(self.pairs, self.hydrogens)
        ):
            indices1 = select_atoms_flat_unique(
                dl.first_frame_chem, smarts1, hydrogens=h1
            )
            indices2 = select_atoms_flat_unique(
                dl.first_frame_chem, smarts2, hydrogens=h2
            )

            mol_ids1 = molecule_ids[indices1]
            mol_ids2 = molecule_ids[indices2]

            exclude_self = smarts1 == smarts2
            if exclude_self:
                print(
                    f"Pair {pair_idx}: Found {len(indices1)} atoms. All intramolecular interactions will be excluded."
                )
            else:
                print(
                    f"Pair {pair_idx}: Found {len(indices1)} atoms for group 1 and {len(indices2)} for group 2."
                )

            all_pair_data.append(
                (
                    jnp.array(indices1),
                    jnp.array(indices2),
                    jnp.array(mol_ids1),
                    jnp.array(mol_ids2),
                    exclude_self,
                )
            )

            # Create structure visualization for this pair
            img = visualize_selected_molecules(dl.first_frame_chem, indices1, indices2)
            if img is not None:
                # Create safe filename from SMARTS patterns
                safe_smarts1 = smarts1.replace("[", "").replace("]", "").replace(":", "_")
                safe_smarts2 = smarts2.replace("[", "").replace("]", "").replace(":", "_")
                path = self.figures / f"{safe_smarts1}_{safe_smarts2}.png"
                idx = 0
                while path.exists():
                    path = self.figures / f"{safe_smarts1}_{safe_smarts2}_{idx}.png"
                    idx += 1
                img.save(path)
                print(f"Saved structure visualization for pair {pair_idx}: {path}")

        ## --------------------------------------------------------------------
        ## Step 2: Process trajectory to identify bonds in each frame
        ## --------------------------------------------------------------------
        print(f"\nStep 2: Processing {dl.total_frames} frames to identify bonds...")
        all_bond_matrices = []
        with tqdm(total=dl.total_frames, desc="Identifying bonds") as pbar:
            for batch in dl:
                positions_batch, cell_batch = (
                    jnp.array(batch["position"]),
                    jnp.array(batch["cell"]),
                )

                for i in range(positions_batch.shape[0]):
                    positions_frame = positions_batch[i]
                    cell_frame = cell_batch[i] if cell_batch.ndim == 4 else cell_batch

                    frame_bonds = []
                    for (
                        indices1,
                        indices2,
                        mol_ids1,
                        mol_ids2,
                        exclude_self,
                    ) in all_pair_data:
                        bond_matrix = _compute_bond_matrix(
                            positions_frame,
                            cell_frame,
                            indices1,
                            indices2,
                            mol_ids1,
                            mol_ids2,
                            self.distance_cutoff,
                            exclude_self,
                        )
                        frame_bonds.append(bond_matrix)
                    all_bond_matrices.append(frame_bonds)
                    pbar.update(1)

        ## --------------------------------------------------------------------
        ## Step 3: Calculate autocorrelation and lifetimes
        ## --------------------------------------------------------------------
        print("\nStep 3: Calculating autocorrelation and lifetimes...")
        final_results = {}
        for pair_idx, _ in enumerate(self.pairs):
            bond_history = jnp.stack([frame[pair_idx] for frame in all_bond_matrices])
            num_frames = bond_history.shape[0]
            max_delay = num_frames // 2

            h_avg = jnp.mean(bond_history)
            if h_avg == 0:
                print(f"Warning: No bonds ever formed for pair {pair_idx}. Skipping.")
                autocorr, lifetime_ps = np.zeros(max_delay), 0.0
            else:
                print(f"Calculating autocorrelation for pair {pair_idx} using FFT...")
                n_a, n_b = bond_history.shape[1], bond_history.shape[2]

                # Perform convolution of the signal with its time-reversed self
                conv_result = fftconvolve(
                    bond_history, bond_history[::-1, :, :], mode="full", axes=0
                )

                # Extract sums for positive lags and sum over atom pairs
                autocorr_sum_vs_lag = jnp.sum(conv_result, axis=(1, 2))[
                    num_frames - 1 : num_frames - 1 + max_delay
                ]

                # Normalize by the number of samples at each lag
                num_elements_at_lag_t = (num_frames - jnp.arange(max_delay)) * n_a * n_b
                mean_product_vs_lag = autocorr_sum_vs_lag / num_elements_at_lag_t
                autocorr = mean_product_vs_lag / h_avg

                autocorr = np.array(autocorr)
                lifetime_ps = np.sum(autocorr) * time_step_fs / 1000.0

            time_axis_ps = (np.arange(max_delay) * time_step_fs) / 1000.0
            final_results[pair_idx] = {
                "autocorrelation": autocorr.tolist(),
                "lifetime_integrated_ps": lifetime_ps.item(),
                "time_axis_ps": time_axis_ps.tolist(),
            }
            print(f"Pair {pair_idx} Average Lifetime (Integrated): {lifetime_ps:.3f} ps")

        self.results = final_results
        self._plot_results(self.results)
