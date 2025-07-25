import zntrack
import numpy as np
from scipy.integrate import simpson
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt
from pathlib import Path

class KirkwoodBuffIntegral(zntrack.Node):
    """
    Calculate the Kirkwood-Buff integral
    Using finite size correction from DOI: 10.1103/PhysRevE.97.051301
    """
    rdf_data: dict[str, list[float]] = zntrack.deps() # Idealy a rdf with very long ranges is needed!
    dr: float = zntrack.params() # r spacing in the rdf


    results: dict = zntrack.outs()
    figures: Path = zntrack.outs_path(zntrack.nwd / "figures")
    def run(self):
        self.results = {}
        self.figures.mkdir(parents=True, exist_ok=True)
        for pair, g_r in self.rdf_data.items():
            g_r = jnp.array(g_r)
            L = len(g_r)*self.dr
            jnp.arange(len(g_r))
            r = jnp.arange(0, L, self.dr)
            h_r = g_r - 1
            x = r/L

            w = 4*np.pi * r**2 * (1 - 3/2*x + 1/2*x**3)
            u2 = w * (1 + 3/2*x + 9/4*x**2)

            kbi = jsp.integrate.trapezoid(y = h_r * u2, x = r)
            self.results[pair] = float(kbi)

            plt.figure()
            plt.plot(r, h_r * u2, label=r"$h(r) u_2(r)$")
            plt.xlabel("r [Å]")
            plt.ylabel("Integrand")
            plt.title(f"KBI integrand for {pair}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(self.figures / f"{pair}_kbi_integrand.png")
            plt.close()