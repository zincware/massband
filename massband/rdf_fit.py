import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.special import erf, erfc

log = logging.getLogger(__name__)


def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def skewed_gaussian(x, amp, mu, sigma, alpha):
    norm = (x - mu) / (np.sqrt(2) * sigma)
    return amp * np.exp(-(norm**2)) * (1 + erf(alpha * norm))


def emg(x, amp, mu, sigma, lam, c):
    arg1 = lam / 2 * (2 * mu + lam * sigma**2 - 2 * x)
    arg2 = (mu + lam * sigma**2 - x) / (np.sqrt(2) * sigma)
    return amp * 0.5 * lam * np.exp(arg1) * erfc(arg2) + c


def find_first_peak_fit(
    r: np.ndarray,
    g_r: np.ndarray,
    smoothing_sigma: float = 2.0,
    fit_method: str = "emg",  # 'gaussian', 'skewed_gaussian', 'emg', 'none'
):
    g_r_smooth = gaussian_filter1d(g_r, sigma=smoothing_sigma)

    start_idx = np.argmax(g_r_smooth > 1.0)
    peak_idx = start_idx + np.argmax(g_r_smooth[start_idx:])

    window = 10
    i_min = max(0, peak_idx - window)
    i_max = min(len(r), peak_idx + window)
    r_fit = r[i_min:i_max]
    g_fit = g_r_smooth[i_min:i_max]

    fit_curve = np.zeros_like(r)
    r_peak = r[peak_idx]
    sigma_peak = np.nan
    popt = []
    pcov = None

    try:
        if fit_method == "gaussian":
            p0 = [g_fit.max(), r[peak_idx], 0.2]
            popt, pcov = curve_fit(gaussian, r_fit, g_fit, p0=p0)
            fit_curve = gaussian(r, *popt)
            r_peak = popt[1]
            sigma_peak = np.sqrt(np.diag(pcov))[1]

        elif fit_method == "skewed_gaussian":
            p0 = [g_fit.max(), r[peak_idx], 0.2, 1.0]
            popt, pcov = curve_fit(skewed_gaussian, r_fit, g_fit, p0=p0)
            fit_curve = skewed_gaussian(r, *popt)

            # Numerically locate the peak of the fitted curve
            r_peak = r[np.argmax(fit_curve)]
            sigma_peak = np.sqrt(np.diag(pcov))[
                1
            ]  # Optional: still use uncertainty on mu

        elif fit_method == "emg":
            p0 = [g_fit.max(), r[peak_idx], 0.2, 1.0, 0.95]
            bounds = ([0, 0, 0, 0, 0], [10, 10, 2, 10, 2])
            popt, pcov = curve_fit(emg, r_fit, g_fit, p0=p0, bounds=bounds, maxfev=10000)
            fit_curve = emg(r, *popt)
            r_peak = popt[1]
            sigma_peak = np.sqrt(np.diag(pcov))[1]

        elif fit_method == "none":
            # Just return smoothed peak, no fitting
            r_peak = r[peak_idx]
            sigma_peak = 0.0
            fit_curve = np.zeros_like(r)

        else:
            raise ValueError(f"Unknown fit_method '{fit_method}'.")

    except Exception as e:
        log.warning(f"Fitting failed with method '{fit_method}': {e}")
        r_peak = r[peak_idx]
        sigma_peak = np.nan

    return {
        "r_peak": r_peak,
        "r_peak_uncertainty": sigma_peak,
        "fit_curve": fit_curve,
        "params": popt,
        "covariance": pcov,
        "smoothed_rdf": g_r_smooth,
    }


def plot_rdf(
    rdfs: defaultdict,
    save_path: Path,
    bin_width: float = 0.1,
    smoothing_sigma: float = 2.0,
    fit_method: str = "skewed_gaussian",  # or 'none', 'gaussian', 'skewed_gaussian', 'emg'
):
    n_rdfs = len(rdfs)
    n_cols = 3
    n_rows = (n_rdfs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    for ax, ((label_a, label_b), g_r_list) in zip(axes, rdfs.items()):
        g_r_array = np.stack(g_r_list)
        g_r_mean = np.array(np.mean(g_r_array, axis=0))
        r = np.array(0.5 * (np.arange(len(g_r_mean)) + 0.5) * bin_width)

        peak_fit = find_first_peak_fit(
            r, g_r_mean, smoothing_sigma=smoothing_sigma, fit_method=fit_method
        )

        ax.plot(r, g_r_mean, label="RDF (raw)", alpha=0.5)
        ax.plot(r, peak_fit["smoothed_rdf"], "--", label="RDF (smoothed)")
        if fit_method != "none":
            ax.plot(r, peak_fit["fit_curve"], color="black", label=f"{fit_method} fit")

        ax.axvline(
            peak_fit["r_peak"],
            color="red",
            linestyle=":",
            label=f"$r_\\mathrm{{peak}}$ = {peak_fit['r_peak']:.2f} ± {peak_fit['r_peak_uncertainty']:.2f} Å",
        )

        ax.set_xlabel("Distance r (Å)")
        ax.set_ylabel("g(r)")
        ax.set_title(f"Radial Distribution Function: {label_a} - {label_b}")
        ax.legend()
        ax.grid(True)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
