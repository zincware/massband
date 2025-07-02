import logging
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.special import erf, erfc
from uravu.relationship import Relationship

log = logging.getLogger(__name__)

FIT_METHODS = Literal[
    "none",
    "gaussian",
    "skewed_gaussian",
    "emg",
    "generalized_gaussian",
    "skewed_generalized_gaussian",
]


def gaussian(x: np.ndarray, amp: float, mu: float, sigma: float) -> np.ndarray:
    """Standard Gaussian function."""
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def generalized_gaussian(
    x: np.ndarray, amp: float, mu: float, sigma: float, beta: float
) -> np.ndarray:
    """Generalized Gaussian function."""
    return amp * np.exp(-(np.abs((x - mu) / sigma) ** beta))


def skewed_gaussian(
    x: np.ndarray, amp: float, mu: float, sigma: float, alpha: float
) -> np.ndarray:
    """Skewed Gaussian function."""
    norm_val = (x - mu) / (np.sqrt(2) * sigma)
    return amp * np.exp(-(norm_val**2)) * (1 + erf(alpha * norm_val))


def smooth_skewed_generalized_gaussian(
    x: np.ndarray, amp: float, mu: float, sigma: float, beta: float, alpha: float
) -> np.ndarray:
    """
    Skewed Generalized Gaussian function.

    The asymmetry in the scale is implemented using a sign-dependent correction.
    """
    # Modify the scale by a factor (1 + alpha * sign) where sign is computed pointwise
    factor = 1 + alpha * np.sign(x - mu)
    norm_val = (x - mu) / (sigma * factor)
    return amp * np.exp(-(np.abs(norm_val) ** beta))


def emg(
    x: np.ndarray, amp: float, mu: float, sigma: float, lam: float, c: float
) -> np.ndarray:
    """Exponentially Modified Gaussian function."""
    arg1 = lam / 2 * (2 * mu + lam * sigma**2 - 2 * x)
    arg2 = (mu + lam * sigma**2 - x) / (np.sqrt(2) * sigma)
    return amp * 0.5 * lam * np.exp(arg1) * erfc(arg2) + c


@dataclass
class PeakFitResult:
    r_peak: float = np.nan
    r_peak_uncertainty: float = np.nan
    fit_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    lower_ci: np.ndarray = field(default_factory=lambda: np.array([]))
    upper_ci: np.ndarray = field(default_factory=lambda: np.array([]))
    params: Optional[np.ndarray] = None
    covariance: Optional[np.ndarray] = None
    smoothed_rdf: np.ndarray = field(default_factory=lambda: np.array([]))
    success: bool = False


@dataclass
class BayesianFitResult:
    samples: np.ndarray = field(default_factory=lambda: np.array([]))
    peak_positions: np.ndarray = field(default_factory=lambda: np.array([]))
    fit_curves: np.ndarray = field(default_factory=lambda: np.array([]))
    r_peak: float = np.nan
    r_peak_uncertainty: float = np.nan
    lower_ci: np.ndarray = field(default_factory=lambda: np.array([]))
    upper_ci: np.ndarray = field(default_factory=lambda: np.array([]))
    success: bool = False


# --- UTILITY FUNCTIONS ---


def find_peak_window_by_gradient(
    r: np.ndarray,
    g_r_smooth: np.ndarray,
    min_threshold: float = 1.0,
    second_derivative_threshold: float = 1,
    window_size: float = 5.0,
    min_pts: int = 5,
) -> Tuple[int, int]:
    """
    Find adaptive window around RDF peak using second derivative (curvature)
    to detect shoulders or inflection points.
    """
    dr = r[1] - r[0]
    max_pts = int(window_size / dr)
    min_pts = max(min_pts, 2)

    # Start at first index above threshold
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
        log.debug(f"Left: second derivative at index {i_min} = {curv:.4f}")
        if curv > second_derivative_threshold:
            log.debug(
                f"Left: curvature exceeds threshold {second_derivative_threshold} at index {i_min}"
            )
            break

    # Right expansion
    i_max = peak_idx
    while i_max < len(r) - 2 and (i_max - peak_idx) < max_pts:
        i_max += 1
        if (i_max - peak_idx) < min_pts:
            continue
        curv = second_deriv[i_max]
        log.debug(f"Right: second derivative at index {i_max} = {curv:.4f}")
        if curv > second_derivative_threshold:
            log.debug(
                f"Right: curvature exceeds threshold {second_derivative_threshold} at index {i_max}"
            )
            break

    i_min = max(0, i_min)
    i_max = min(len(r) - 1, i_max)

    log.debug(
        f"Peak window found: indices {i_min} to {i_max} (r = {r[i_min]:.2f} Å to {r[i_max]:.2f} Å)"
    )
    return i_min, i_max


def find_peak_window(
    r: np.ndarray, g_r: np.ndarray, min_threshold: float = 1.0, window_scale: float = 0.5
) -> Tuple[int, int]:
    """Find a reasonable window around the first peak."""
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
    fit_func: Callable,
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

        peak_std = np.std(peak_positions)
        lower_peak, upper_peak = np.percentile(
            peak_positions, [50 * (1 - ci), 50 * (1 + ci)]
        )
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
    fit_func: Callable,
    bounds: Tuple[Tuple[float, float], ...],
    n_samples: int = 2000,
    ci: float = 0.65,
) -> BayesianFitResult:
    """Perform Bayesian fitting using uravu."""
    result = BayesianFitResult()
    try:
        modeller = Relationship(
            fit_func,
            r,
            g_r,
            bounds=bounds,
            ordinate_error=g_r * 0.1,
        )
        modeller.mcmc()
        samples = np.array([v.samples for v in modeller.variables]).T

        peak_positions = []
        fit_curves = []
        for sample in samples:
            curve = fit_func(r, *sample)
            peak_idx = np.argmax(curve)
            peak_positions.append(r[peak_idx])
            fit_curves.append(curve)

        fit_curves = np.array(fit_curves)
        result.samples = samples
        result.peak_positions = np.array(peak_positions)
        result.fit_curves = fit_curves
        result.r_peak = np.median(peak_positions)
        result.r_peak_uncertainty = np.std(peak_positions)
        result.lower_ci = np.percentile(fit_curves, 50 * (1 - ci), axis=0)
        result.upper_ci = np.percentile(fit_curves, 50 * (1 + ci), axis=0)
        result.success = True

    except Exception as e:
        log.error(f"Bayesian fitting failed: {e}")
        result.success = False

    return result


# --- MAIN PEAK FITTING FUNCTION ---


def fit_first_peak(  # noqa: C901
    r: np.ndarray,
    g_r: np.ndarray,
    fit_method: str,
    bayesian: bool = False,
    smoothing_sigma: float = 2.0,
    min_threshold: float = 1.0,
    window_scale: float = 0.5,
    ci: float = 0.65,
    n_samples: int = 1000,
) -> PeakFitResult:
    """Find and optionally fit the first peak in RDF data and return the result as a dataclass."""
    result = PeakFitResult(smoothed_rdf=gaussian_filter1d(g_r, sigma=smoothing_sigma))

    try:
        i_min, i_max = find_peak_window_by_gradient(r, result.smoothed_rdf, min_threshold)
        r_fit = r[i_min:i_max]
        g_fit = g_r[i_min:i_max]
        g_smooth_fit = result.smoothed_rdf[i_min:i_max]

        peak_idx = np.argmax(g_smooth_fit)
        r_peak_guess = r_fit[peak_idx]
        amp_guess = g_smooth_fit[peak_idx]
        sigma_guess = 0.2

        if fit_method == "none":
            # Return smoothed peak location without fitting
            result.r_peak = r_peak_guess
            result.r_peak_uncertainty = 0.0
            result.success = True
            return result  # ⬅️ Early return

        # --- Bayesian fitting ---
        if bayesian:
            func_map = {
                "gaussian": gaussian,
                "skewed_gaussian": skewed_gaussian,
                "emg": emg,
                "generalized_gaussian": generalized_gaussian,
                "skewed_generalized_gaussian": smooth_skewed_generalized_gaussian,
            }
            fit_func = func_map.get(fit_method)
            if fit_func is None:
                raise ValueError("Invalid fitting method")

            if fit_method == "gaussian":
                bounds = (
                    (0, max(g_fit) * 1.2),
                    (min(r_fit), max(r_fit)),
                    (0, max(r_fit) - min(r_fit)),
                )
            elif fit_method == "skewed_gaussian":
                bounds = (
                    (0, max(g_fit)),
                    (min(r_fit), max(r_fit)),
                    (0, max(r_fit) - min(r_fit)),
                    (-10, 10),
                )
            elif fit_method == "emg":
                bounds = (
                    (0, max(g_fit)),
                    (min(r_fit), max(r_fit)),
                    (0, max(r_fit) - min(r_fit)),
                    (0, 10),
                    (0, 2),
                )
            elif fit_method == "generalized_gaussian":
                bounds = (
                    (0, max(g_fit)),
                    (min(r_fit), max(r_fit)),
                    (0, max(r_fit) - min(r_fit)),
                    (2, 5),
                )
            elif fit_method == "skewed_generalized_gaussian":
                bounds = (
                    (0, max(g_fit)),
                    (min(r_fit), max(r_fit)),
                    (0, max(r_fit) - min(r_fit)),
                    (2, 5),
                    (-10, 10),
                )
            else:
                raise ValueError("Invalid fitting method")

            bayes_result = bayesian_fit_uravu(
                r_fit, g_fit, fit_func, bounds, n_samples, ci
            )

            if bayes_result.success:
                median_params = np.median(bayes_result.samples, axis=0)
                result.fit_curve = fit_func(r, *median_params)
                result.r_peak = bayes_result.r_peak
                result.r_peak_uncertainty = bayes_result.r_peak_uncertainty
                result.lower_ci = bayes_result.lower_ci
                result.upper_ci = bayes_result.upper_ci
                result.params = median_params
                result.covariance = np.cov(bayes_result.samples.T)
                result.success = True
            else:
                raise RuntimeError("Bayesian fitting failed")

        # --- Classic curve fitting ---
        else:
            func_map = {
                "gaussian": gaussian,
                "skewed_gaussian": skewed_gaussian,
                "emg": emg,
                "generalized_gaussian": generalized_gaussian,
                "skewed_generalized_gaussian": smooth_skewed_generalized_gaussian,
            }
            fit_func = func_map.get(fit_method)
            if fit_func is None:
                raise ValueError("Invalid fitting method")

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
            else:
                raise ValueError("Invalid fitting method")

            popt, pcov = curve_fit(fit_func, r_fit, g_fit, p0=p0, bounds=bounds)
            result.fit_curve = fit_func(r, *popt)
            uncertainty, lower_ci, upper_ci = calculate_peak_uncertainty(
                r, fit_func, popt, pcov, n_samples, ci
            )

            result.r_peak = popt[1]
            result.r_peak_uncertainty = uncertainty
            result.lower_ci = lower_ci
            result.upper_ci = upper_ci
            result.params = popt
            result.covariance = pcov
            result.success = True

    except ValueError as ve:
        log.warning(f"Peak finding failed: {ve}")
    except Exception as e:
        log.error(f"An unexpected error occurred during fitting: {e}")

    return result
