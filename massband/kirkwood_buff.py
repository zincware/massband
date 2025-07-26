from pathlib import Path

import jax.numpy as jnp
import jax.scipy as jsp
import pandas as pd
import matplotlib.pyplot as plt
import zntrack
import logging

log = logging.getLogger(__name__)

import jax.numpy as jnp
import jax.scipy as jsp
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging

log = logging.getLogger(__name__)


def compute_kirkwood_buff_integrals(rdf_data: dict[str, list[float]], dr: float, figures: Path):
    """
    Calculate Kirkwood-Buff integrals from RDF data using the finite size correction
    from DOI: 10.1103/PhysRevE.97.051301

    Parameters
    ----------
        rdf_data: Dict of RDF curves, keyed by "A|B"
        dr: Bin width in Å
        figures: Path to save plot

    Returns
    ----------
        G: pd.DataFrame of Kirkwood-Buff integrals
        results: dict of integrals keyed by "A|B"
    """
    results = {}

    pairs = list(rdf_data.keys())
    structures = sorted({s for pair in pairs for s in pair.split("|")})
    G = pd.DataFrame(0.0, index=structures, columns=structures)

    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1)
    ax.set_ylabel(r"$h(r) u_2(r)$")
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

        w = 4 * jnp.pi * r**2 * (1 - 3 / 2 * x + 1 / 2 * x**3)
        u2 = w * (1 + 3 / 2 * x + 9 / 4 * x**2)

        kbi = jsp.integrate.trapezoid(y=h_r * u2, x=r)
        G.loc[s1, s2] = kbi
        G.loc[s2, s1] = kbi
        results[pair] = float(kbi)

        ax.plot(r, h_r * u2, label=f"{pair}")

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



class KirkWoodBuffAnalyis(zntrack.Node):
    rdf_data: dict = zntrack.deps()
    partial_densities: dict = zntrack.deps()
    dr: float = zntrack.params()

    T: float = zntrack.params() # Temperature in K

    G: pd.DataFrame = zntrack.outs() # G_ij has units A^3/molecule
    results: dict = zntrack.metrics()
    figures: Path = zntrack.outs_path(zntrack.nwd / "figures")

    def run(self):
        kBT = self.T * 8.617e-5 # eV
        total_rho = jnp.sum(jnp.array(self.partial_densities.values())) # molecules/A^3
        rho = jnp.array(self.partial_densities.values())
        log.info("Starting Kirkwood-Buff Analysis")
        print(self.partial_densities)

        self.G, self.results structures = compute_kirkwood_buff_integrals(
            rdf_data=self.rdf_data,
            dr=self.dr,
            figures=self.figures
        )
        log.info("Kirkwood-Buff analysis complete")

        #isothermal_compress = 1/kBT * (jnp.sum([[rho[i]*rho[j]*G[i, j]+total_rho] for i in range(len(structures))] for j in range(len(structures))))