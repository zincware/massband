from massband.diffusion.kinisi_diffusion import KinisiSelfDiffusion
import zntrack
import scipp as sc
from kinisi.arrhenius import Arrhenius
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class KinisiArrhenius(zntrack.Node):
    diff: list[KinisiSelfDiffusion] = zntrack.deps()
    temperatures: list[float] = zntrack.params()
    figures_path: Path = zntrack.outs_path(zntrack.nwd / "figures")
    activation_energy_bound: tuple[float, float] = zntrack.params()
    pre_exponential_factor_bound: tuple[float, float] = zntrack.params()

    activation_energy: dict[str, float] = zntrack.metrics()
    pre_exponential_factor: dict[str, float] = zntrack.metrics()

    def run(self):
        self.figures_path.mkdir(parents=True, exist_ok=True)
        self.activation_energy = {}
        self.pre_exponential_factor = {}

        D = {
            "mean": [x.diffusion["[B-](F)(F)(F)F"]["mean"] for x in self.diff],
            "var": [x.diffusion["[B-](F)(F)(F)F"]["var"] for x in self.diff],
        }

        td = sc.DataArray(
            data=sc.array(
                dims=["temperature"],
                values=D["mean"],
                variances=D["var"],
                unit=sc.Unit("cm^2/s"),
            ),  # hard coded for now
            coords={
                "temperature": sc.Variable(
                    dims=["temperature"], values=self.temperatures, unit="K"
                )
            },
        )

        s = Arrhenius(
            td,
            bounds=[
                [
                    self.activation_energy_bound[0] * sc.Unit("eV"),
                    self.activation_energy_bound[1] * sc.Unit("eV"),
                ],
                [
                    self.pre_exponential_factor_bound[0] * td.data.unit,
                    self.pre_exponential_factor_bound[1] * td.data.unit,
                ],
            ],
        )
        s.mcmc()
        self.activation_energy["[B-](F)(F)(F)F"] = {
            "mean": sc.mean(s.activation_energy).value,
            "std": sc.std(s.activation_energy, ddof=1).value,
        }
        self.pre_exponential_factor["[B-](F)(F)(F)F"] = {
            "mean": sc.mean(s.preexponential_factor).value,
            "std": sc.std(s.preexponential_factor, ddof=1).value,
        }

        credible_intervals = [[16, 84], [2.5, 97.5], [0.15, 99.85]]
        alpha = [0.6, 0.4, 0.2]

        plt.errorbar(
            1000 / td.coords["temperature"].values,
            td.data.values,
            np.sqrt(td.data.variances),
            marker="o",
            ls="",
            color="k",
            zorder=10,
        )

        for i, ci in enumerate(credible_intervals):
            plt.fill_between(
                1000 / td.coords["temperature"].values,
                *np.percentile(s.distribution, ci, axis=1),
                alpha=alpha[i],
                color="#0173B2",
                lw=0,
            )
        plt.yscale("log")
        plt.xlabel("$1000T^{-1}$/K$^{-1}$")
        plt.ylabel("$D$/cm$^2$s$^{-1}$")
        plt.show()
        plt.savefig(self.figures_path / "arrhenius.png", dpi=300, bbox_inches="tight")
