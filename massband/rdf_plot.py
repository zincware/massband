import logging
from pathlib import Path
from typing import DefaultDict, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Import the fitting functions and dataclass result
from massband.rdf_fit import PeakFitResult, find_peak_window_by_gradient, fit_first_peak

log = logging.getLogger(__name__)


def plot_rdf(
    rdfs: DefaultDict[Tuple[str, str], np.ndarray],
    save_path: Path,
    bin_width: float = 0.1,
    smoothing_sigma: float = 2.0,
    bayesian: bool = True,
    fit_method: str = "gaussian",
    min_threshold: float = 1.0,
    window_scale: float = 0.50,
    ci: float = 0.65,
    n_samples: int = 1000,
):
    """Plot RDFs with peak fitting and uncertainty visualization."""
    n_rdfs = len(rdfs)
    n_cols = min(3, n_rdfs)
    n_rows = (n_rdfs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), squeeze=False)
    axes = axes.flatten()

    for ax, ((label_a, label_b), g_r) in zip(axes, rdfs.items()):
        r = 0.5 * (np.arange(len(g_r)) + 0.5) * bin_width

        # Safety check for extreme values
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(g_r)):
            log.warning(
                f"Found non-finite values in RDF data for {label_a} - {label_b}. Skipping."
            )
            continue

        if len(r) > 50000:  # Prevent too many data points
            log.warning(
                f"RDF data for {label_a} - {label_b} has {len(r)} points. Downsampling."
            )
            downsample_factor = len(r) // 10000
            r = r[::downsample_factor]
            g_r = g_r[::downsample_factor]

        try:
            peak_fit: PeakFitResult = fit_first_peak(
                r,
                g_r,
                fit_method=fit_method,
                bayesian=bayesian,
                smoothing_sigma=smoothing_sigma,
                min_threshold=min_threshold,
                window_scale=window_scale,
                ci=ci,
                n_samples=n_samples,
            )
        except Exception as e:
            log.error(f"Error fitting RDF for {label_a} - {label_b}: {e}")
            continue

        ax.plot(r, g_r, label="RDF (raw)", alpha=0.5)

        # Visualize the fit window
        try:
            i_min, i_max = find_peak_window_by_gradient(
                r, peak_fit.smoothed_rdf, min_threshold
            )
        except Exception as e:
            log.error(f"Error finding peak window for {label_a} - {label_b}: {e}")
            continue

        if fit_method != "none" and peak_fit.success:
            ax.plot(
                r[i_min:i_max],
                peak_fit.fit_curve[i_min:i_max],
                color="black",
                label=f"{fit_method} fit",
            )

            if bayesian:
                lower_ci = np.full_like(r, np.nan)
                upper_ci = np.full_like(r, np.nan)
                lower_ci[i_min:i_max] = peak_fit.lower_ci
                upper_ci[i_min:i_max] = peak_fit.upper_ci
                ax.fill_between(
                    r,
                    lower_ci,
                    upper_ci,
                    color="gray",
                    alpha=0.3,
                    label=f"{int(ci * 100)}% CI",
                )
            else:
                ax.plot(r, peak_fit.lower_ci, "k--", alpha=0.3)
                ax.plot(
                    r, peak_fit.upper_ci, "k--", alpha=0.3, label=f"{int(ci * 100)}% CI"
                )

            ax.axvline(
                peak_fit.r_peak,
                color="red",
                linestyle=":",
                label=f"$r_\\mathrm{{peak}}$ = {peak_fit.r_peak:.2f} ± {peak_fit.r_peak_uncertainty:.2f} Å",
            )
            peak_region = (
                peak_fit.r_peak - peak_fit.r_peak_uncertainty,
                peak_fit.r_peak + peak_fit.r_peak_uncertainty,
            )
            ax.axvspan(
                peak_region[0],
                peak_region[1],
                color="red",
                alpha=0.1,
                label="Peak region",
            )

        ax.set_xlabel("Distance r (Å)")
        ax.set_ylabel("g(r)")
        ax.set_title(f"RDF: {label_a} - {label_b}")
        ax.legend(loc="lower left", fontsize="small")

        # Set reasonable axis limits
        ax.set_xlim(0, min(np.max(r), 20))  # Cap x-axis at 20 Å
        ax.set_ylim(0, min(np.max(g_r) * 1.1, 10))  # Cap y-axis at reasonable values
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))  # 1 Å major ticks
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))  # 0.2 Å minor ticks
        ax.yaxis.set_major_locator(
            ticker.MultipleLocator(0.5)
        )  # 0.5 major ticks for g(r)
        ax.yaxis.set_minor_locator(
            ticker.MultipleLocator(0.1)
        )  # 0.1 minor ticks for g(r)
        ax.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.8)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.3)

    for ax in axes[len(rdfs) :]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
