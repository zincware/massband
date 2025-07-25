import zntrack
import numpy as np
from scipy.integrate import simpson




class KirkwoodBuffAnalysis(zntrack.Node):
    """
    Calculate the Kirkwood–Buff integral
    """
    rdf: dict = zntrack.deps() # Idealy a rdf with very long ranges is needed!
    dr: float = zntrack.params() # r spacing in the rdf

    results: dict = zntrack.outs()

    def run(self):
        self.results = {}
        for pair, vals in self.rdf.items():
            vals = np.array(vals)
            r = np.linspace(0, len(vals) * self.dr, len(vals))
            kb = 4*np.pi*(simpson((vals-1)*r**2, dx=self.dr))
            self.results[pair] = kb
        #TODO: Add compressibility calcualtion