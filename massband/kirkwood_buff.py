import logging
from pathlib import Path

import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt
import pandas as pd
import zntrack
import numpy as np
from scipy.constants import physical_constants

log = logging.getLogger(__name__)


def compute_kirkwood_buff_integrals(
    rdf_data: dict[str, list[float]], dr: float, figures: Path
):
    """
    Calculate Kirkwood-Buff integrals from RDF data using the finite size correction
    from DOI: 10.1103/PhysRevE.97.051301 (Eq. 1).

    Parameters
    ----------
        rdf_data: Dict of RDF curves, keyed by "A|B"
        dr: Bin width in Å
        figures: Path to save plot

    Returns
    ----------
        G: pd.DataFrame of Kirkwood-Buff integrals (units: Å^3/molecule)
        results: dict of integrals keyed by "A|B"
        structures: list of unique structure names sorted alphabetically
    """
    results = {}

    pairs = list(rdf_data.keys())
    structures = sorted({s for pair in pairs for s in pair.split("|")})
    G = pd.DataFrame(0.0, index=structures, columns=structures)

    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1)
    ax.set_ylabel(r"$h(r) w(r) 4\pi r^2$")
    ax.set_xlabel("r in Å")
    ax.grid(True)

    log.info("Iterating over RDF pairs")

    for pair, g_r in rdf_data.items():
        s1, s2 = pair.split("|")

        g_r = jnp.array(g_r)
        L = len(g_r) * dr
        r = jnp.arange(0, L, dr)
        h_r = g_r - 1
        x = r / L

        # Finite size correction weighting factor w(r) as per Eq. 20 of 10.1103/PhysRevE.97.051301
        weighting_factor = 1 - 3 / 2 * x + 1 / 2 * x**3
        integrand_value = h_r * weighting_factor * 4 * jnp.pi * r**2
        kbi = jsp.integrate.trapezoid(y=integrand_value, x=r)

        G.loc[s1, s2] = kbi
        G.loc[s2, s1] = kbi
        results[pair] = float(kbi)

        ax.plot(r, integrand_value, label=f"{pair}")

    fig.suptitle("KBI integrand")
    fig.legend(
        bbox_to_anchor=(0, 0, 1, 0),
        loc="center",
        ncol=4,
        fontsize=12,
        frameon=False,
    )
    fig.tight_layout()
    plt.savefig(figures / "kbi_integrand.png", bbox_inches="tight")
    plt.close()

    return G, results, structures


def calculate_observables(G_df: pd.DataFrame, partial_densities: dict, T: float):
    """
    Calculate macroscopic observables from Kirkwood-Buff integrals and partial densities.

    Parameters
    ----------
    G_df : pd.DataFrame
        DataFrame of Kirkwood-Buff integrals. Units: Å^3/molecule.
    partial_densities : dict
        Dictionary of partial densities (molecules/A^3) keyed by structure name.
    T : float
        Temperature in Kelvin.

    Returns
    -------
    dict
        A dictionary containing the calculated observables with consistent units.
    """
    structures = list(G_df.index)
    rho = jnp.array([partial_densities[s] for s in structures])  # Units: molecules/A^3
    G_mat = jnp.array(G_df.values)  # Units: Å^3/molecule

    kB_J_per_K = physical_constants["Boltzmann constant"][0]  # J/K
    kBT_J = T * kB_J_per_K  # Units: Joules

    total_rho = jnp.sum(rho)  # Units: molecules/A^3

    observables = {}

    # Chemical Potential Derivatives (dmu_i/drho_j)
    K_matrix = jnp.diag(rho) + jnp.outer(rho, rho) * G_mat

    try:
        K_inv = jnp.linalg.inv(K_matrix)  # Units: A^3/molecules
        # dmu_i / d_rho_j = kBT * (K_inv)_ij
        # Units: J * (A^3/molecules) = J*A^3/molecules
        dmu_drho = kBT_J * K_inv
        observables["dmu_drho"] = pd.DataFrame(
            jnp.array(dmu_drho), index=structures, columns=structures
        )
    except np.linalg.LinAlgError:
        log.error(
            "Singular K_matrix for chemical potential derivatives. Check densities/integrals."
        )
        observables["dmu_drho"] = "Calculation failed: Singular matrix"

    # Partial Molar Volumes (V_i)
    M_matrix = (
        jnp.eye(len(structures)) + G_mat * rho[None, :]
    )  # M_ij = delta_ij + G_ij * rho_j

    try:
        M_inv = jnp.linalg.inv(M_matrix)  # Units: Dimensionless
        partial_molar_volumes = (
            jnp.sum(M_inv, axis=1) / total_rho
        )  # Units: (Dimensionless) / (molecules/A^3) = A^3/molecule
        observables["partial_molar_volumes"] = pd.Series(
            np.array(partial_molar_volumes), index=structures
        )
    except np.linalg.LinAlgError:
        log.error(
            "Singular M_matrix for partial molar volumes. Check densities/integrals."
        )
        observables["partial_molar_volumes"] = "Calculation failed: Singular matrix"

    # Isothermal Compressibility (beta_T)

    try:
        det_K = jnp.linalg.det(K_matrix)
        det_rho_diag = jnp.linalg.det(jnp.diag(rho))

        if det_rho_diag != 0:
            dimensionless_factor = det_K / det_rho_diag
            isothermal_compressibility_per_molecule_J = dimensionless_factor / (
                kBT_J * total_rho
            )
            isothermal_compressibility_A3_per_J = (
                isothermal_compressibility_per_molecule_J
            )
            # convert to Pa^-1
            isothermal_compressibility_Pa_inv = (
                isothermal_compressibility_A3_per_J * 1e-30
            )

            observables["isothermal_compressibility"] = float(
                isothermal_compressibility_Pa_inv
            )
            observables["isothermal_compressibility_A3_per_J"] = float(
                isothermal_compressibility_A3_per_J
            )
        else:
            log.error(
                "Determinant of diagonal rho matrix is zero. Cannot calculate compressibility."
            )
            observables["isothermal_compressibility"] = (
                "Calculation failed: Zero density"
            )
            observables["isothermal_compressibility_A3_per_J"] = (
                "Calculation failed: Zero density"
            )

    except np.linalg.LinAlgError:
        log.error(
            "Singular K_matrix for compressibility calculation. Check densities/integrals."
        )
        observables["isothermal_compressibility"] = (
            "Calculation failed: Singular matrix"
        )
        observables["isothermal_compressibility_A3_per_J"] = (
            "Calculation failed: Singular matrix"
        )

    observables["kBT_J"] = float(kBT_J)
    observables["total_density"] = float(total_rho)

    return observables


class KirkwoodBuffAnalysis(zntrack.Node):
    """Calculate Macroscopic Observables
    Following DOI: 10.1063/1.2943318 (for general relations) and standard KB theory.

    Parameters
    ----------
    rdf_data : dict
        RDF curves, keyed by "A|B".
    partial_densities : dict
        Partial number densities of each component in molecules/Å^3, keyed by component name.
    dr : float
        Bin width for RDF data in Å.
    T : float
        Temperature in Kelvin.
    """

    rdf_data: dict = zntrack.deps()
    partial_densities: dict = zntrack.deps()
    dr: float = zntrack.params(0.05)  # Angstroms

    T: float = zntrack.params(300)  # Temperature in K

    results: dict = zntrack.metrics()
    figures: Path = zntrack.outs_path(zntrack.nwd / "figures")

    def run(self):
        log.info("Starting Kirkwood-Buff Analysis")

        # G in Å^3/molecule
        G, kbi_results, structures = compute_kirkwood_buff_integrals(
            rdf_data=self.rdf_data, dr=self.dr, figures=self.figures
        )

        observables = calculate_observables(G, self.partial_densities, self.T)

        self.results = {
            "kbi_integrals": kbi_results,  # Å^3/molecule
            "dmu_drho": observables["dmu_drho"].to_dict(),  # J*A^3/molecule
            "partial_molar_volumes": observables[
                "partial_molar_volumes"
            ].to_dict(),  # A^3/molecule
            "isothermal_compressibility_Pa_inv": observables[
                "isothermal_compressibility"
            ],  # Pa^-1 (m^3/J)
            "isothermal_compressibility_A3_per_J": observables[
                "isothermal_compressibility_A3_per_J"
            ],  # A^3/J
            "kBT_J": observables["kBT_J"],  # Joules
            "total_density": observables["total_density"],  # molecules/A^3
        }

        log.info("Kirkwood-Buff analysis complete")
        log.info(
            f"Chemical Potential Derivatives (J*A^3/molecule): \n{observables['dmu_drho']}"
        )
        log.info(
            f"Partial Molar Volumes (A^3/molecule): \n{observables['partial_molar_volumes']}"
        )
        log.info(
            f"Isothermal Compressibility (Pa^-1): {observables['isothermal_compressibility']}"
        )
