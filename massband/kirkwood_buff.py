import zntrack
import numpy as np
from scipy.integrate import simpson


class KirkwoodBuffAnalysis(zntrack.Node):
    """
    Calculate the Kirkwood–Buff integral
    """
    rdf: dict = zntrack.deps()
    bin_width: float = zntrack.params()

    resutls: dict = zntrack.outs()

    def run(self):
        self.resutls = {}
        for pair, vals in self.rdf.items():
            vals = np.array(vals)
            r = np.linspace(0, len(vals) * self.bin_width, len(vals))
            kb = 4*np.pi*(simpson((vals-1)*r**2, dx=self.bin_width))
            self.resutls[pair] = kb
        #TODO: Add compressibility calcualtion