import typing as t

import ase
import plotly.graph_objects as go

FIGURES = dict[str, go.Figure]
FRAMES = list[ase.Atoms]


class ComparisonResults(t.TypedDict):
    frames: FRAMES
    figures: FIGURES
