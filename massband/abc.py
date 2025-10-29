import typing as t

import matplotlib.figure
import plotly.graph_objects as go

FIGURES = dict[str, go.Figure | matplotlib.figure.Figure]


class ComparisonResults(t.TypedDict):
    figures: FIGURES
