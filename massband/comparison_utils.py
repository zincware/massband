"""Utility functions for comparing node results."""

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def interpolate_to_common_grid(
    x_arrays: list[np.ndarray], y_arrays: list[np.ndarray]
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Interpolate multiple datasets to a common x-axis grid.

    Parameters
    ----------
    x_arrays : list[np.ndarray]
        List of x-coordinate arrays
    y_arrays : list[np.ndarray]
        List of y-coordinate arrays

    Returns
    -------
    x_common : np.ndarray
        Common x-axis grid
    y_interpolated : list[np.ndarray]
        List of y-values interpolated to common grid
    """
    # Find common x-range
    x_min = max(x[0] for x in x_arrays)
    x_max = min(x[-1] for x in x_arrays)

    # Create common grid using finest resolution
    n_points = max(len(x) for x in x_arrays)
    x_common = np.linspace(x_min, x_max, n_points)

    # Interpolate all y-arrays to common grid
    y_interpolated = []
    for x, y in zip(x_arrays, y_arrays):
        y_interp = np.interp(x_common, x, y)
        y_interpolated.append(y_interp)

    return x_common, y_interpolated


def compute_statistics(
    data1: np.ndarray,
    data2: np.ndarray,
    uncertainty1: np.ndarray | None = None,
    uncertainty2: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute statistical differences between two datasets.

    Parameters
    ----------
    data1 : np.ndarray
        First dataset
    data2 : np.ndarray
        Second dataset
    uncertainty1 : np.ndarray | None
        Uncertainty in first dataset
    uncertainty2 : np.ndarray | None
        Uncertainty in second dataset

    Returns
    -------
    dict[str, float]
        Dictionary containing RMSE, MAE, max_deviation, and correlation
    """
    # Handle NaN values
    valid_mask = np.isfinite(data1) & np.isfinite(data2)

    if not np.any(valid_mask):
        return {
            "rmse": np.nan,
            "mae": np.nan,
            "max_deviation": np.nan,
            "correlation": np.nan,
        }

    d1 = data1[valid_mask]
    d2 = data2[valid_mask]

    diff = d1 - d2

    stats = {
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mae": float(np.mean(np.abs(diff))),
        "max_deviation": float(np.max(np.abs(diff))),
        "correlation": float(np.corrcoef(d1, d2)[0, 1]) if len(d1) > 1 else np.nan,
    }

    return stats


def create_overlay_plot(
    x_data: list[np.ndarray],
    y_data: list[np.ndarray],
    labels: list[str],
    title: str,
    xlabel: str,
    ylabel: str,
    uncertainties: list[np.ndarray | None] | None = None,
    use_plotly: bool = False,
) -> go.Figure | plt.Figure:
    """Create overlay plot comparing multiple datasets.

    Parameters
    ----------
    x_data : list[np.ndarray]
        List of x-coordinate arrays
    y_data : list[np.ndarray]
        List of y-coordinate arrays
    labels : list[str]
        Labels for each dataset
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    uncertainties : list[np.ndarray | None] | None
        List of uncertainty arrays for each dataset
    use_plotly : bool
        If True, returns plotly figure; otherwise matplotlib

    Returns
    -------
    go.Figure | plt.Figure
        Comparison plot
    """
    if uncertainties is None:
        uncertainties = [None] * len(x_data)

    if use_plotly:
        fig = go.Figure()
        for x, y, label, unc in zip(x_data, y_data, labels, uncertainties):
            fig.add_trace(
                go.Scatter(x=x, y=y, mode="lines", name=label, line={"width": 2})
            )

            if unc is not None:
                # Add uncertainty band
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([x, x[::-1]]),
                        y=np.concatenate([y + unc, (y - unc)[::-1]]),
                        fill="toself",
                        fillcolor="rgba(0,100,80,0.2)",
                        line={"color": "rgba(255,255,255,0)"},
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

        fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel)
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        for x, y, label, unc in zip(x_data, y_data, labels, uncertainties):
            ax.plot(x, y, label=label, linewidth=2)

            if unc is not None:
                ax.fill_between(x, y - unc, y + unc, alpha=0.3)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig


def create_difference_plot(
    x_data: list[np.ndarray],
    y_data: list[np.ndarray],
    labels: list[str],
    title: str,
    xlabel: str,
    ylabel: str,
    reference_idx: int = 0,
    use_plotly: bool = False,
) -> go.Figure | plt.Figure:
    """Create difference plot showing deviations from reference.

    Parameters
    ----------
    x_data : list[np.ndarray]
        List of x-coordinate arrays
    y_data : list[np.ndarray]
        List of y-coordinate arrays
    labels : list[str]
        Labels for each dataset
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label (for differences)
    reference_idx : int
        Index of reference dataset
    use_plotly : bool
        If True, returns plotly figure; otherwise matplotlib

    Returns
    -------
    go.Figure | plt.Figure
        Difference plot
    """
    # Interpolate to common grid
    x_common, y_interpolated = interpolate_to_common_grid(x_data, y_data)
    y_ref = y_interpolated[reference_idx]

    if use_plotly:
        fig = go.Figure()
        for i, (y, label) in enumerate(zip(y_interpolated, labels)):
            if i == reference_idx:
                continue
            diff = y - y_ref
            fig.add_trace(
                go.Scatter(
                    x=x_common,
                    y=diff,
                    mode="lines",
                    name=f"{label} - {labels[reference_idx]}",
                )
            )

        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel)
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, (y, label) in enumerate(zip(y_interpolated, labels)):
            if i == reference_idx:
                continue
            diff = y - y_ref
            ax.plot(x_common, diff, label=f"{label} - {labels[reference_idx]}")

        ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig


def create_bar_comparison(
    values: dict[str, list[float]],
    errors: dict[str, list[float]] | None,
    labels: list[str],
    title: str,
    ylabel: str,
    use_plotly: bool = False,
) -> go.Figure | plt.Figure:
    """Create grouped bar chart comparing values across datasets.

    Parameters
    ----------
    values : dict[str, list[float]]
        Dictionary mapping category names to lists of values (one per dataset)
    errors : dict[str, list[float]] | None
        Dictionary mapping category names to error bars
    labels : list[str]
        Dataset labels
    title : str
        Plot title
    ylabel : str
        Y-axis label
    use_plotly : bool
        If True, returns plotly figure; otherwise matplotlib

    Returns
    -------
    go.Figure | plt.Figure
        Bar comparison plot
    """
    if errors is None:
        errors = {k: [0] * len(v) for k, v in values.items()}

    categories = list(values.keys())
    n_datasets = len(labels)

    if use_plotly:
        fig = go.Figure()
        for i, label in enumerate(labels):
            vals = [values[cat][i] for cat in categories]
            errs = [errors[cat][i] for cat in categories]
            fig.add_trace(
                go.Bar(
                    name=label,
                    x=categories,
                    y=vals,
                    error_y={"type": "data", "array": errs},
                )
            )

        fig.update_layout(title=title, yaxis_title=ylabel, barmode="group")
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(categories))
        width = 0.8 / n_datasets

        for i, label in enumerate(labels):
            vals = [values[cat][i] for cat in categories]
            errs = [errors[cat][i] for cat in categories]
            offset = (i - n_datasets / 2) * width + width / 2
            ax.bar(x + offset, vals, width, label=label, yerr=errs, capsize=5)

        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        return fig
