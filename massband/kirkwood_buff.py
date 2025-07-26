from pathlib import Path

import jax.numpy as jnp
import jax.scipy as jsp
import pandas as pd
import matplotlib.pyplot as plt
import zntrack
import logging

log = logging.getLogger(__name__)

class KirkwoodBuffIntegral(zntrack.Node):
    """
    Calculate the Kirkwood-Buff integral
    Using finite size correction from DOI: 10.1103/PhysRevE.97.051301
    """

    rdf_data: dict[str, list[float]] = (
        zntrack.deps()
    )  # Ideally a rdf with very long ranges is needed!
    dr: float = zntrack.params()  # r spacing in the rdf

    results: dict = zntrack.outs()
    G: pd.DataFrame = zntrack.outs()
    figures: Path = zntrack.outs_path(zntrack.nwd / "figures")

    def run(self):
        self.results = {}

        pairs = list(self.rdf_data.keys())
        structures = sorted({s for pair in pairs for s in pair.split("|")})
        self.G = pd.DataFrame(0.0, index=structures, columns=structures)

        self.figures.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1)
        ax.set_ylabel(r"$h(r) u_2(r)$")
        ax.set_xlabel("r in Å")
        ax.grid(True)

        log.info("Iterating over Pairs")

        for pair, g_r in self.rdf_data.items():
            s1, s2 = pair.split("|")

            g_r = jnp.array(g_r)
            L = len(g_r) * self.dr
            r = jnp.arange(0, L, self.dr)
            h_r = g_r - 1
            x = r / L

            w = 4 * jnp.pi * r**2 * (1 - 3 / 2 * x + 1 / 2 * x**3)
            u2 = w * (1 + 3 / 2 * x + 9 / 4 * x**2)

            kbi = jsp.integrate.trapezoid(y=h_r * u2, x=r)
            self.G.loc[s1, s2] = kbi
            self.G.loc[s2, s1] = kbi
            self.results[pair] = float(kbi)

            ax.plot(r, h_r * u2, label=f"{pair}")

        print(self.G)
        fig.suptitle("KBI integrand")
        fig.legend(
            bbox_to_anchor=(0, 0, 1, 0),
            loc="center",
            ncol=4,
            fontsize=12,
            frameon=False,
        )
        fig.tight_layout()
        log.info("Saving KBI results")
        plt.savefig(self.figures / "kbi_integrand.png", bbox_inches="tight")
        plt.close()


class KirkWoodBuffAnalyis(zntrack.Node):
    g: dict = zntrack.deps()
    partial_densities: dict = zntrack.deps()

    def run(self):
        log.info("Starting Kirkwood-Buff Analysis")
        print(self.partial_densities)