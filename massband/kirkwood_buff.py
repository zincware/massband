from pathlib import Path

import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt
import zntrack


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
    figures: Path = zntrack.outs_path(zntrack.nwd / "figures")

    def run(self):
        self.results = {}
        self.figures.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1)
        ax.set_ylabel(r"$h(r) u_2(r)$")
        ax.set_xlabel("r in Å")
        ax.grid(True)

        for pair, g_r in self.rdf_data.items():
            g_r = jnp.array(g_r)
            L = len(g_r) * self.dr
            r = jnp.arange(0, L, self.dr)
            h_r = g_r - 1
            x = r / L

            w = 4 * jnp.pi * r**2 * (1 - 3 / 2 * x + 1 / 2 * x**3)
            u2 = w * (1 + 3 / 2 * x + 9 / 4 * x**2)

            kbi = jsp.integrate.trapezoid(y=h_r * u2, x=r)
            self.results[pair] = float(kbi)

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
        plt.savefig(self.figures / "kbi_integrand.png", bbox_inches="tight")
        plt.close()
