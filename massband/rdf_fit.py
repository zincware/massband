import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.special import erf, erfc
from scipy.stats import norm
from uravu.distribution import Distribution
from uravu.relationship import Relationship

log = logging.getLogger(__name__)


def gaussian(x: np.ndarray, amp: float, mu: float, sigma: float) -> np.ndarray:
    """Standard Gaussian function."""
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def generalized_gaussian(x, amp, mu, sigma, beta):
    """Generalized Gaussian function."""
    return amp * np.exp(-(np.abs((x - mu) / sigma) ** beta))


def skewed_gaussian(
    x: np.ndarray, amp: float, mu: float, sigma: float, alpha: float
) -> np.ndarray:
    """Skewed Gaussian function."""
    norm = (x - mu) / (np.sqrt(2) * sigma)
    return amp * np.exp(-(norm**2)) * (1 + erf(alpha * norm))


# def smooth_skewed_generalized_gaussian(x, amp, mu, sigma, beta, alpha):
#     core = np.exp(-np.abs((x - mu) / sigma)**beta)
#     skew = 1 + erf(alpha * (x - mu))
#     return amp * core * skew


def smooth_skewed_generalized_gaussian(x, amp, mu, sigma, beta, alpha):
    """
    Skewed Generalized Gaussian function.

    Parameters:
        x (array-like): Input values.
        amp (float): Amplitude (height of the peak).
        mu (float): Center position.
        sigma (float): Scale parameter (controls width).
        beta (float): Shape parameter (controls tail sharpness).
                      beta=2 -> Gaussian, beta<2 = heavier tails, beta>2 = sharper peak.
        alpha (float): Skewness parameter.
                       alpha > 0 = right-skewed, alpha < 0 = left-skewed, 0 = symmetric.

    Returns:
        np.ndarray: Function values.
    """
    norm = (x - mu) / (sigma * (1 + alpha * np.sign(x - mu)))
    return amp * np.exp(-(np.abs(norm) ** beta))


# TODO: skewed_generialized_gaussian


def emg(
    x: np.ndarray, amp: float, mu: float, sigma: float, lam: float, c: float
) -> np.ndarray:
    """Exponentially Modified Gaussian function."""
    arg1 = lam / 2 * (2 * mu + lam * sigma**2 - 2 * x)
    arg2 = (mu + lam * sigma**2 - x) / (np.sqrt(2) * sigma)
    return amp * 0.5 * lam * np.exp(arg1) * erfc(arg2) + c


import numpy as np


def find_peak_window_by_gradient(
    r: np.ndarray,
    g_r_smooth: np.ndarray,
    min_threshold: float = 1.0,
    second_derivative_threshold: float = 1,
    window_size: float = 5.0,
    min_pts: int = 5,
) -> tuple[int, int]:
    """
    Find adaptive window around RDF peak using second derivative (curvature)
    to detect shoulders or inflection points.

    Parameters:
    - r: Distance array (uniform spacing).
    - g_r_smooth: Smoothed RDF.
    - min_threshold: Minimum g(r) to define the start of a peak.
    - second_derivative_threshold: Stop expanding window if curvature drops below this.
    - window_size: Max total window size (Å).
    - min_pts: Minimum distance in points to allow expansion.

    Returns:
    - (i_min, i_max): Indices of the window around the peak.
    """
    dr = r[1] - r[0]
    max_pts = int(window_size / dr)
    min_pts = max(min_pts, 2)

    # Start after min threshold
    start_idx = np.argmax(g_r_smooth > min_threshold)
    if start_idx == 0 and g_r_smooth[0] <= min_threshold:
        raise ValueError("No peak above threshold found")

    peak_idx = start_idx + np.argmax(g_r_smooth[start_idx:])
    first_deriv = np.gradient(g_r_smooth, dr)
    second_deriv = np.gradient(first_deriv, dr)

    # Left expansion
    i_min = peak_idx
    while i_min > 1 and (peak_idx - i_min) < max_pts:
        i_min -= 1
        if (peak_idx - i_min) < min_pts:
            continue
        curv = second_deriv[i_min]
        print(f"Left: second derivative at index {i_min} = {curv:.4f}")
        if curv > second_derivative_threshold:
            print(
                f"Left: curvature dropped below {second_derivative_threshold}, stopping at {i_min}"
            )
            break

    # Right expansion
    i_max = peak_idx
    while i_max < len(r) - 2 and (i_max - peak_idx) < max_pts:
        i_max += 1
        if (i_max - peak_idx) < min_pts:
            continue
        curv = second_deriv[i_max]
        print(f"Right: second derivative at index {i_max} = {curv:.4f}")
        if curv > second_derivative_threshold:
            print(
                f"Right: curvature dropped below {second_derivative_threshold}, stopping at {i_max}"
            )
            break

    i_min = max(0, i_min)
    i_max = min(len(r) - 1, i_max)

    print(
        f"Peak window found: {i_min} to {i_max} for peak at index {peak_idx} "
        f"({r[i_min]:.2f} Å to {r[i_max]:.2f} Å)"
    )

    return i_min, i_max


def find_peak_window(
    r: np.ndarray, g_r: np.ndarray, min_threshold: float = 1.0, window_scale: float = 0.5
) -> tuple[int, int]:
    """Find reasonable window around the first peak."""
    start_idx = np.argmax(g_r > min_threshold)
    if start_idx == 0 and g_r[0] <= min_threshold:
        raise ValueError("No peak above threshold found")

    peak_idx = start_idx + np.argmax(g_r[start_idx:])

    window_size = int(window_scale / (r[1] - r[0])) if len(r) > 1 else len(r)
    i_min = max(0, peak_idx - window_size)
    i_max = min(len(r), peak_idx + window_size)

    return i_min, i_max


def calculate_peak_uncertainty(
    r: np.ndarray,
    fit_func: callable,
    popt: np.ndarray,
    pcov: Optional[np.ndarray],
    n_samples: int = 1000,
    ci: float = 0.65,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Calculate peak position uncertainty with confidence intervals."""
    if pcov is None:
        return np.nan, np.zeros_like(r), np.zeros_like(r)

    try:
        samples = np.random.multivariate_normal(popt, pcov, n_samples)
        peak_positions = []
        fit_curves = []

        for params in samples:
            fit_curve = fit_func(r, *params)
            peak_idx = np.argmax(fit_curve)
            peak_positions.append(r[peak_idx])
            fit_curves.append(fit_curve)

        # Calculate CI for peak position
        peak_std = np.std(peak_positions)
        lower_peak, upper_peak = np.percentile(
            peak_positions, [50 * (1 - ci), 50 * (1 + ci)]
        )

        # Calculate CI for fit curves
        fit_curves = np.array(fit_curves)
        lower_ci = np.percentile(fit_curves, 50 * (1 - ci), axis=0)
        upper_ci = np.percentile(fit_curves, 50 * (1 + ci), axis=0)

        return peak_std, lower_ci, upper_ci

    except Exception as e:
        log.warning(f"Uncertainty calculation failed: {e}")
        return np.nan, np.zeros_like(r), np.zeros_like(r)


def bayesian_fit_uravu(
    r: np.ndarray,
    g_r: np.ndarray,
    fit_func: callable,
    p0: np.ndarray,
    bounds: Tuple[Tuple[float, float], ...],
    n_samples: int = 2000,
    ci: float = 0.65,
) -> Dict[str, Any]:
    """Perform Bayesian fitting using uravu."""
    try:
        # Create distribution objects for each data point
        y_distributions = [
            Distribution(norm.rvs(loc=y, scale=0.1 * y, size=1000)) for y in g_r
        ]

        # Create relationship model
        modeller = Relationship(
            fit_func,
            r,
            g_r,
            bounds=bounds,
            ordinate_error=g_r * 0.1,  # Assuming 10% error TODO: this is a parameter!!
        )

        # Run MCMC sampling
        modeller.mcmc()

        # Get samples
        samples = np.array([v.samples for v in modeller.variables]).T

        # Calculate peak positions and fit curves
        peak_positions = []
        fit_curves = []
        for sample in samples:
            fit_curve = fit_func(r, *sample)
            peak_idx = np.argmax(fit_curve)
            peak_positions.append(r[peak_idx])
            fit_curves.append(fit_curve)

        fit_curves = np.array(fit_curves)

        return {
            "samples": samples,
            "peak_positions": np.array(peak_positions),
            "fit_curves": fit_curves,
            "r_peak": np.median(peak_positions),
            "r_peak_uncertainty": np.std(peak_positions),
            "lower_ci": np.percentile(fit_curves, 50 * (1 - ci), axis=0),
            "upper_ci": np.percentile(fit_curves, 50 * (1 + ci), axis=0),
            "success": True,
        }
    except Exception as e:
        log.error(f"Bayesian fitting failed: {e}")
        return {"success": False, "r_peak": np.nan, "r_peak_uncertainty": np.nan}


def fit_first_peak(
    r: np.ndarray,
    g_r: np.ndarray,
    fit_method: Literal[
        "gaussian",
        "skewed_gaussian",
        "emg",
        "generalized_gaussian",
        "none",
        "skewed_generalized_gaussian",
    ],
    bayesian: bool = False,
    smoothing_sigma: float = 2.0,
    min_threshold: float = 1.0,
    window_scale: float = 0.5,
    ci: float = 0.65,
    n_samples: int = 1000,
) -> Dict[str, Any]:
    """Find and fit the first peak in RDF data."""
    g_r_smooth = gaussian_filter1d(g_r, sigma=smoothing_sigma)

    result = {
        "r_peak": np.nan,
        "r_peak_uncertainty": np.nan,
        "fit_curve": np.zeros_like(r),
        "lower_ci": np.zeros_like(r),
        "upper_ci": np.zeros_like(r),
        "params": None,
        "covariance": None,
        "smoothed_rdf": g_r_smooth,
        "success": False,
    }

    try:
        # i_min, i_max = find_peak_window(r, g_r_smooth, min_threshold, window_scale)
        i_min, i_max = find_peak_window_by_gradient(r, g_r_smooth, min_threshold)

        r_fit = r[i_min:i_max]
        g_fit = g_r[i_min:i_max]
        g_smooth_fit = g_r_smooth[i_min:i_max]

        # use the smoothed data for initial guesses
        peak_idx = np.argmax(g_smooth_fit)
        r_peak_guess = r_fit[peak_idx]
        amp_guess = g_smooth_fit[peak_idx]
        sigma_guess = 0.2

        if fit_method == "none":
            result.update(
                {"r_peak": r_peak_guess, "r_peak_uncertainty": 0.0, "success": True}
            )
        elif bayesian:
            # TODO: improve the bounds and initial guesses!
            if fit_method == "gaussian":
                sigma_min = 0
                sigma_max = max(r_fit) - min(r_fit)
                bounds = (
                    (0, max(g_fit) * 1.2),  # amp
                    (min(r_fit), max(r_fit)),  # mu
                    (sigma_min, sigma_max),  # sigma
                )
                bayes_result = bayesian_fit_uravu(
                    r_fit, g_fit, gaussian, None, bounds, n_samples, ci
                )
            elif fit_method == "skewed_gaussian":
                bounds = (
                    (0, max(g_fit)),
                    (min(r_fit), max(r_fit)),
                    (0, max(r_fit) - min(r_fit)),
                    (-10, 10),
                )
                bayes_result = bayesian_fit_uravu(
                    r_fit, g_fit, skewed_gaussian, None, bounds, n_samples, ci
                )
            elif fit_method == "emg":
                bounds = (
                    (0, max(g_fit)),
                    (min(r_fit), max(r_fit)),
                    (0, max(r_fit) - min(r_fit)),
                    (0, 10),
                    (0, 2),
                )
                bayes_result = bayesian_fit_uravu(
                    r_fit, g_fit, emg, None, bounds, n_samples, ci
                )
            elif fit_method == "generalized_gaussian":
                bounds = (
                    (0, max(g_fit)),
                    (min(r_fit), max(r_fit)),
                    (0, max(r_fit) - min(r_fit)),
                    (2, 5),
                )
                bayes_result = bayesian_fit_uravu(
                    r_fit, g_fit, generalized_gaussian, None, bounds, n_samples, ci
                )
            elif fit_method == "skewed_generalized_gaussian":
                bounds = (
                    (0, max(g_fit)),
                    (min(r_fit), max(r_fit)),
                    (0, max(r_fit) - min(r_fit)),
                    (2, 5),
                    (-10, 10),
                )
                bayes_result = bayesian_fit_uravu(
                    r_fit,
                    g_fit,
                    smooth_skewed_generalized_gaussian,
                    None,
                    bounds,
                    n_samples,
                    ci,
                )

            if bayes_result["success"]:
                # Evaluate median parameters on full r
                median_params = np.median(bayes_result["samples"], axis=0)
                fit_func = {
                    "gaussian": gaussian,
                    "skewed_gaussian": skewed_gaussian,
                    "emg": emg,
                    "generalized_gaussian": generalized_gaussian,
                    "skewed_generalized_gaussian": smooth_skewed_generalized_gaussian,
                }[fit_method]

                fit_curve = fit_func(r, *median_params)

                result.update(
                    {
                        "r_peak": bayes_result["r_peak"],
                        "r_peak_uncertainty": bayes_result["r_peak_uncertainty"],
                        "fit_curve": fit_curve,
                        "lower_ci": np.percentile(
                            [fit_func(r, *s) for s in bayes_result["samples"]],
                            50 * (1 - ci),
                            axis=0,
                        ),
                        "upper_ci": np.percentile(
                            [fit_func(r, *s) for s in bayes_result["samples"]],
                            50 * (1 + ci),
                            axis=0,
                        ),
                        "params": median_params,
                        "covariance": np.cov(bayes_result["samples"].T),
                        "success": True,
                    }
                )
            else:
                raise RuntimeError("Bayesian fitting failed")

        else:
            fit_funcs = {
                "gaussian": gaussian,
                "skewed_gaussian": skewed_gaussian,
                "emg": emg,
                "generalized_gaussian": generalized_gaussian,
                "skewed_generalized_gaussian": smooth_skewed_generalized_gaussian,
            }
            fit_func = fit_funcs[fit_method]

            if fit_method == "gaussian":
                p0 = [amp_guess, r_peak_guess, sigma_guess]
                bounds = ([0, min(r_fit), 0], [np.inf, max(r_fit), np.inf])
            elif fit_method == "skewed_gaussian":
                p0 = [amp_guess, r_peak_guess, sigma_guess, 1.0]
                bounds = (
                    [0, min(r_fit), 0, -np.inf],
                    [np.inf, max(r_fit), np.inf, np.inf],
                )
            elif fit_method == "emg":
                p0 = [amp_guess, r_peak_guess, sigma_guess, 1.0, 1.0]
                bounds = (
                    [0, min(r_fit), 0, 0, 0],
                    [np.inf, max(r_fit), np.inf, np.inf, np.inf],
                )
            elif fit_method == "generalized_gaussian":
                p0 = [amp_guess, r_peak_guess, sigma_guess, 2.0]
                bounds = ([0, min(r_fit), 0, 2], [np.inf, max(r_fit), np.inf, 5])
            elif fit_method == "skewed_generalized_gaussian":
                p0 = [amp_guess, r_peak_guess, sigma_guess, 2.0, 1.0]
                bounds = (
                    [0, min(r_fit), 0, 2, -np.inf],
                    [np.inf, max(r_fit), np.inf, 5, np.inf],
                )

            popt, pcov = curve_fit(fit_func, r_fit, g_fit, p0=p0, bounds=bounds)
            fit_curve = fit_func(r, *popt)

            # Calculate uncertainty with CI
            uncertainty, lower_ci, upper_ci = calculate_peak_uncertainty(
                r, fit_func, popt, pcov, n_samples, ci
            )

            result.update(
                {
                    "r_peak": popt[1]
                    if fit_method != "skewed_gaussian"
                    else r[np.argmax(fit_curve)],
                    "r_peak_uncertainty": uncertainty,
                    "fit_curve": fit_curve,
                    "lower_ci": lower_ci,
                    "upper_ci": upper_ci,
                    "params": popt,
                    "covariance": pcov,
                    "success": True,
                }
            )

    except ValueError as e:
        log.warning(f"Peak finding failed: {e}")
    except Exception as e:
        log.error(f"An unexpected error occurred during fitting: {e}")

    return result


def plot_rdf(
    rdfs: defaultdict,
    save_path: Path,
    bin_width: float = 0.1,
    smoothing_sigma: float = 2.0,
    bayesian: bool = True, # use MCMC fitting to get a better estimate of the peak
    fit_method: Literal[
        "gaussian",
        "skewed_gaussian",
        "emg",
        "generalized_gaussian",
        "skewed_generalized_gaussian",
        "none",
    ] = "gaussian",
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

    for ax, ((label_a, label_b), g_r_list) in zip(axes, rdfs.items()):
        g_r_array = np.stack(g_r_list)
        g_r_mean = np.mean(g_r_array, axis=0)
        r = 0.5 * (np.arange(len(g_r_mean)) + 0.5) * bin_width

        try:
            peak_fit = fit_first_peak(
                r,
                g_r_mean,
                fit_method=fit_method,
                bayesian=bayesian,
                smoothing_sigma=smoothing_sigma,
                min_threshold=min_threshold,
                window_scale=window_scale,
                ci=ci,
                n_samples=n_samples,
            )
        except Exception as e:
            print(f"Error fitting RDF for {label_a} - {label_b}: {e}")
            peak_fit = None

        ax.plot(r, g_r_mean, label="RDF (raw)", alpha=0.5)
        # ax.plot(r, peak_fit["smoothed_rdf"], "--", label="RDF (smoothed)")

        # Highlighted changes start here - adding fit window visualization
        if peak_fit is not None and fit_method != "none" and peak_fit["success"]:
            # Get the fit window indices from the peak fitting function
            # i_min, i_max = find_peak_window(r, peak_fit["smoothed_rdf"], min_threshold, window_scale)
            i_min, i_max = find_peak_window_by_gradient(
                r, peak_fit["smoothed_rdf"], min_threshold
            )

            # plot vertical span for the fit window
            # ax.axvspan(r[i_min], r[i_max-1], color='cyan', alpha=0.1, label='Fit window')

            ax.plot(
                r[i_min:i_max],
                peak_fit["fit_curve"][i_min:i_max],
                color="black",
                label=f"{fit_method} fit",
            )

            # Plot confidence interval
            if bayesian:
                ax.fill_between(
                    r,
                    peak_fit["lower_ci"],
                    peak_fit["upper_ci"],
                    color="gray",
                    alpha=0.3,
                    label=f"{int(ci * 100)}% CI",
                )
            else:
                ax.plot(r, peak_fit["lower_ci"], "k--", alpha=0.3)
                ax.plot(
                    r,
                    peak_fit["upper_ci"],
                    "k--",
                    alpha=0.3,
                    label=f"{int(ci * 100)}% CI",
                )

            ax.axvline(
                peak_fit["r_peak"],
                color="red",
                linestyle=":",
                label=f"$r_\\mathrm{{peak}}$ = {peak_fit['r_peak']:.2f} ± {peak_fit['r_peak_uncertainty']:.2f} Å",
            )
            # shade the peak region
            peak_region = (
                peak_fit["r_peak"] - peak_fit["r_peak_uncertainty"],
                peak_fit["r_peak"] + peak_fit["r_peak_uncertainty"],
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
        # Set minor ticks every 0.1 units on x-axis
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

        # Optional: set minor ticks on y-axis too (e.g. every 0.1)
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

        ax.grid(True)
        ax.grid(
            True, which="minor", linestyle=":", linewidth=0.5, alpha=0.3
        )  # Lighter minor grid


    for ax in axes[len(rdfs) :]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
