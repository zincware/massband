from collections import defaultdict
from pathlib import Path

import ase.io
import jax.numpy as jnp
import matplotlib.pyplot as plt
import zntrack
from scipy.signal import correlate
from tqdm import tqdm

from massband.dataloader import TimeBatchedLoader


class RadiusOfGyration(zntrack.Node):
    """Calculate the radius of gyration for molecules in a trajectory."""

    file: str | Path = zntrack.deps_path()
    structures: list[str] = zntrack.params(default_factory=list)
    figures: Path = zntrack.outs_path(zntrack.nwd / "figures")
    results: dict = zntrack.metrics()
    data: dict = zntrack.outs()

    traj: Path = zntrack.outs_path(zntrack.nwd / "traj")

    def run(self) -> None:
        dl = TimeBatchedLoader(
            file=self.file,
            batch_size=64,
            structures=self.structures,
            wrap=False,
            com=False,
            properties=["position", "masses"],
        )
        self.results = {}
        self.data = {}

        results = defaultdict(list)

        self.traj.mkdir(parents=True, exist_ok=True)
        atoms_buffer = defaultdict(list)  # Collect strings to write per molecule type

        for batch_output in tqdm(dl, desc="Calculating Radius of Gyration"):
            pos = batch_output["position"]
            masses_dict = batch_output["masses"]
            for key in pos:
                # Get atom positions for all molecules, shape: (batch, n_mols, n_atoms, 3)
                molecule_positions = pos[key]
                # Get masses for this structure - since com=False, we get individual atom masses
                atom_masses = masses_dict[key]  # shape: (total_atoms_for_structure,)
                
                # Get indices for all molecules of this type to reshape masses properly
                mol_indices = jnp.array(dl.indices[key])  # shape: (n_mols, n_atoms_per_mol)
                # Reshape atom masses to match molecule structure: (n_mols, n_atoms_per_mol)
                molecule_masses = atom_masses.reshape(mol_indices.shape)
                
                # Reshape positions to match: (batch, n_mols, n_atoms_per_mol, 3)
                molecule_positions = molecule_positions.reshape(
                    -1, *molecule_masses.shape, 3
                )
                # raise ValueError(molecule_masses.shape, molecule_positions.shape)

                # print(f"Processing {key} with {molecule_positions.shape} frames and {molecule_positions.shape[1]} molecules")

                # --- Vectorized Rg Calculation ---
                # Total mass for each molecule, shape: (n_mols,)
                total_mass_per_mol = jnp.sum(molecule_masses, axis=1)

                # Center of mass for each molecule in each frame
                # Shape: (batch, n_mols, 3)
                center_of_mass = (
                    jnp.sum(
                        molecule_masses[None, :, :, None] * molecule_positions, axis=2
                    )
                    / total_mass_per_mol[None, :, None]
                )

                # Displacements from CoM, shape: (batch, n_mols, n_atoms, 3)
                displacements = molecule_positions - center_of_mass[:, :, None, :]

                # Squared distances, shape: (batch, n_mols, n_atoms)
                squared_distances = jnp.sum(displacements**2, axis=3)

                # Mass-weighted sum of squared distances, shape: (batch, n_mols)
                sum_mass_weighted_sq_dist = jnp.sum(
                    molecule_masses[None, :, :] * squared_distances, axis=2
                )

                rg_squared = sum_mass_weighted_sq_dist / total_mass_per_mol[None, :]

                # Final Rg for this batch, shape: (batch, n_mols)
                rg = jnp.sqrt(rg_squared)

                results[key].append(rg)

        self.figures.mkdir(parents=True, exist_ok=True)
        for key in atoms_buffer:
            # Write the collected ASE Atoms objects to a file
            ase.io.write(
                self.traj / f"{key}.xyz",
                atoms_buffer[key],
                format="extxyz",
                append=True,
            )

        # --- Per-Molecule Analysis and Plotting ---
        # This will store the mean Rg time series for global analysis
        mean_rg_timeseries_all_types = {}
        # This will store the number of molecules for weighted averaging
        num_molecules_per_type = {}

        for smiles, rg_values_list in results.items():
            # 1. Aggregate results from all batches for the current molecule type
            # Shape -> (total_frames, n_molecules_of_this_type)
            if not rg_values_list:
                continue
            rg_values = jnp.concatenate(rg_values_list, axis=0)

            num_molecules = rg_values.shape[1]
            num_molecules_per_type[smiles] = num_molecules

            # 2. Calculate statistics
            # Mean and Std Dev at each frame, averaged across molecules (axis=1)
            mean_rg_per_frame = jnp.mean(rg_values, axis=1)
            std_rg_per_frame = jnp.std(rg_values, axis=1)

            # Overall mean and std dev across all frames and all molecules
            overall_mean_rg = jnp.mean(rg_values)
            overall_std_rg = jnp.std(rg_values)

            # Store for global analysis
            mean_rg_timeseries_all_types[smiles] = mean_rg_per_frame

            # 3. Save results and raw data
            self.results[smiles] = {
                "mean": float(overall_mean_rg),
                "std": float(overall_std_rg),
            }
            self.data[smiles] = {
                "rg_values": rg_values.flatten().tolist(),
            }

            # 4. Create plots
            filename_smiles = "".join(c for c in smiles if c.isalnum())

            # --- Time-series plot ---
            plt.figure(figsize=(10, 6))
            x_frames = jnp.arange(len(mean_rg_per_frame))
            plt.plot(x_frames, mean_rg_per_frame, label="Mean Rg per Frame")
            plt.fill_between(
                x_frames,
                mean_rg_per_frame - std_rg_per_frame,
                mean_rg_per_frame + std_rg_per_frame,
                alpha=0.2,
                label="Std Dev across molecules",
            )
            plt.xlabel("Frame")
            plt.ylabel("Radius of Gyration (Å)")
            plt.title(f"Radius of Gyration for {smiles}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(self.figures / f"rog_{filename_smiles}.png", dpi=300)
            plt.close()

            # --- Histogram ---
            plt.figure(figsize=(10, 6))
            plt.hist(rg_values.flatten(), bins=50, density=True, alpha=0.75)
            plt.xlabel("Radius of Gyration (Å)")
            plt.ylabel("Probability Density")
            plt.title(f"Rg Distribution for {smiles}")
            plt.grid(True, alpha=0.3)
            plt.savefig(self.figures / f"hist_rog_{filename_smiles}.png", dpi=300)
            plt.close()

            # --- Autocorrelation ---
            ac = self._autocorrelation(mean_rg_per_frame)
            plt.figure(figsize=(10, 6))
            plt.plot(ac)
            plt.xlabel("Lag (frames)")
            plt.ylabel("Autocorrelation")
            plt.title(f"Autocorrelation of Mean Rg: {smiles}")
            plt.axhline(0, color="grey", linestyle="--", alpha=0.5)
            plt.grid(True, alpha=0.3)
            plt.savefig(self.figures / f"ac_rog_{filename_smiles}.png", dpi=300)
            plt.close()

        # --- Global Rg Analysis ---
        if len(mean_rg_timeseries_all_types) > 1:
            # Calculate a weighted average of the mean Rg time series
            total_weighted_rg_sum = 0
            total_molecules = 0

            # Use the shortest trajectory length for consistent array shapes
            min_frames = min(len(ts) for ts in mean_rg_timeseries_all_types.values())

            for smiles, timeseries in mean_rg_timeseries_all_types.items():
                num_mols = num_molecules_per_type[smiles]
                total_weighted_rg_sum += timeseries[:min_frames] * num_mols
                total_molecules += num_mols

            global_mean_rg_per_frame = total_weighted_rg_sum / total_molecules

            # Calculate overall global mean and std from all flattened data
            all_rg_values_flat = jnp.concatenate(
                [jnp.array(self.data[smiles]["rg_values"]) for smiles in self.data]
            )
            global_mean = jnp.mean(all_rg_values_flat)
            global_std = jnp.std(all_rg_values_flat)

            self.results["global"] = {
                "mean": float(global_mean),
                "std": float(global_std),
            }

            # Plot the global weighted-average time series
            plt.figure(figsize=(10, 6))
            x_frames_global = jnp.arange(len(global_mean_rg_per_frame))
            plt.plot(
                x_frames_global,
                global_mean_rg_per_frame,
                label="Global Mean RG (Weighted)",
            )
            plt.xlabel("Frame")
            plt.ylabel("Radius of Gyration (Å)")
            plt.title("Global Mean Radius of Gyration (Weighted by Molecule Count)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(self.figures / "global_rog.png", dpi=300)
            plt.close()

    def _autocorrelation(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x - jnp.mean(x)
        result = correlate(x, x, mode="full")
        result = result[result.size // 2 :]
        result = result / result[0]
        return result
