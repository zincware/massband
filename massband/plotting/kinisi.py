"""Centralized plotting for kinisi-based analysis."""
import dataclasses
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)


@dataclasses.dataclass
class PlottingConfig:
    """Configuration for plotting kinisi results."""

    displacement_label: str
    displacement_unit: str
    value_label: str
    value_unit: str
    msd_title: str
    msd_filename: str
    ci_title: str
    ci_filename: str
    hist_title: str
    hist_filename: str
    hist_label: str


def _get_text_box(data, config: PlottingConfig, results: Optional[dict] = None) -> str:
    """Generate the text for the annotation box on the histogram."""
    if results and data.structure in results:
        res = results[data.structure]
        # Check for diffusion results keys
        if "diffusion_coefficient" in res and "asymmetric_uncertainty" in res:
            return "\n".join(
                (
                    f"Mean: {res['diffusion_coefficient']:.3e}",
                    f"Std: \u00b1{res['std']:.3e}",
                    f"68% CI: [{res['credible_interval_68'][0]:.3e}, {res['credible_interval_68'][1]:.3e}]",
                    f"95% CI: [{res['credible_interval_95'][0]:.3e}, {res['credible_interval_95'][1]:.3e}]",
                    f"Asymmetric Uncertainty: [{res['asymmetric_uncertainty'][0]:.3e}, {res['asymmetric_uncertainty'][1]:.3e}]",
                )
            )

    # Default for conductivity or if detailed results are not available
    std_val = np.std(data.samples)
    ci68 = np.percentile(data.samples, [16, 84])
    ci95 = np.percentile(data.samples, [2.5, 97.5])
    return "\n".join(
        (
            f"Mean: {data.mean_value:.6f} {config.value_unit}",
            f"Std: \u00b1{std_val:.6f} {config.value_unit}",
            f"68% CI: [{ci68[0]:.6f}, {ci68[1]:.6f}] {config.value_unit}",
            f"95% CI: [{ci95[0]:.6f}, {ci95[1]:.6f}] {config.value_unit}",
        )
    )


def plot_kinisi_results(
    data, data_path: Path, config: PlottingConfig, results: Optional[dict] = None
):
    """Generate and save plots for kinisi-based analysis (diffusion/conductivity)."""
    credible_intervals = [[16, 84], [2.5, 97.5], [0.15, 99.85]]
    alpha = [0.6, 0.4, 0.2]

    # Displacement with std plot
    fig, ax = plt.subplots()
    ax.errorbar(data.dt, data.displacement, data.displacement_std)
    ax.set_ylabel(f"{config.displacement_label}/{config.displacement_unit}")
    ax.set_xlabel(r"$\Delta t$/ps")
    ax.set_title(config.msd_title)
    try:
        fig.savefig(data_path / config.msd_filename, dpi=300)
    except FileNotFoundError:
        log.warning("Could not save plot %s. Does the directory exist?", config.msd_filename)
    plt.close(fig)

    # Displacement with credible intervals plot
    fig, ax = plt.subplots()
    ax.plot(data.dt, data.displacement, "k-")
    for i, ci in enumerate(credible_intervals):
        low, high = np.percentile(data.distribution, ci, axis=1)
        ax.fill_between(data.dt, low, high, alpha=alpha[i], color="#0173B2", lw=0)
    ax.axvline(data.start_dt, c="k", ls="--", label=f"start_dt = {data.start_dt} ps")
    ax.set_ylabel(f"{config.displacement_label}/{config.displacement_unit}")
    ax.set_xlabel(r"$\Delta t$/ps")
    ax.set_title(config.ci_title)
    ax.legend()
    try:
        fig.savefig(data_path / config.ci_filename, dpi=300)
    except FileNotFoundError:
        log.warning("Could not save plot %s. Does the directory exist?", config.ci_filename)
    plt.close(fig)

    # Histogram plot
    fig, ax = plt.subplots()
    ax.hist(
        data.samples,
        density=True,
        bins=50,
        color="lightblue",
        edgecolor="k",
    )
    ax.axvline(data.mean_value, c="red", ls="--", label=config.hist_label)

    ci68 = np.percentile(data.samples, [16, 84])
    ax.axvline(ci68[0], c="blue", ls=":", label="68% CI")
    ax.axvline(ci68[1], c="blue", ls=":")

    ax.set_xlabel(f"{config.value_label}/{config.value_unit}")
    ax.set_ylabel(f"p({config.value_label})/{config.value_unit}")
    ax.set_title(config.hist_title)
    ax.legend()

    textstr = _get_text_box(data, config, results)
    ax.text(
        0.95,
        0.95,
        textstr,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        fontsize=8,
    )

    try:
        fig.savefig(data_path / config.hist_filename, dpi=300)
    except FileNotFoundError:
        log.warning("Could not save plot %s. Does the directory exist?", config.hist_filename)
    plt.close(fig)
