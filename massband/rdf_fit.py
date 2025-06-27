import logging
from collections import defaultdict
from pathlib import Path
from typing import Literal, Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.special import erf, erfc

log = logging.getLogger(__name__)

def gaussian(x: np.ndarray, amp: float, mu: float, sigma: float) -> np.ndarray:
    """Standard Gaussian function."""
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def skewed_gaussian(x: np.ndarray, amp: float, mu: float, sigma: float, alpha: float) -> np.ndarray:
    """Skewed Gaussian function."""
    norm = (x - mu) / (np.sqrt(2) * sigma)
    return amp * np.exp(-(norm**2)) * (1 + erf(alpha * norm))

def emg(x: np.ndarray, amp: float, mu: float, sigma: float, lam: float, c: float) -> np.ndarray:
    """Exponentially Modified Gaussian function."""
    arg1 = lam / 2 * (2 * mu + lam * sigma**2 - 2 * x)
    arg2 = (mu + lam * sigma**2 - x) / (np.sqrt(2) * sigma)
    return amp * 0.5 * lam * np.exp(arg1) * erfc(arg2) + c

def find_peak_window(
    r: np.ndarray,
    g_r: np.ndarray,
    min_threshold: float = 1.0,
    window_scale: float = 0.5
) -> tuple[int, int]:
    """
    Find reasonable window around the first peak.
    
    Args:
        r: Distance array
        g_r: RDF values
        min_threshold: Minimum g(r) value to consider as peak region
        window_scale: Window size in Ångströms
        
    Returns:
        Tuple of (start_index, end_index)
    """
    # Find first point above threshold
    start_idx = np.argmax(g_r > min_threshold)
    if start_idx == 0 and g_r[0] <= min_threshold:
        raise ValueError("No peak above threshold found")
    
    # Find peak within remaining data
    peak_idx = start_idx + np.argmax(g_r[start_idx:])
    r_peak = r[peak_idx]
    
    # Create adaptive window based on Ångström scale
    window_size = int(window_scale / (r[1] - r[0])) if len(r) > 1 else len(r)
    i_min = max(0, peak_idx - window_size)
    i_max = min(len(r), peak_idx + window_size)
    
    return i_min, i_max

def calculate_peak_uncertainty(
    r: np.ndarray,
    fit_func: callable,
    popt: np.ndarray,
    pcov: Optional[np.ndarray],
    n_samples: int = 1000
) -> float:
    """
    Calculate peak position uncertainty using Monte Carlo sampling.
    
    Args:
        r: Distance array
        fit_func: Fitting function
        popt: Optimal parameters
        pcov: Covariance matrix
        n_samples: Number of samples for MC
        
    Returns:
        Peak position uncertainty
    """
    if pcov is None:
        return np.nan
    
    try:
        # Generate parameter samples
        samples = np.random.multivariate_normal(popt, pcov, n_samples)
        
        # Find peak positions for each sample
        peak_positions = []
        for params in samples:
            fit_curve = fit_func(r, *params)
            peak_idx = np.argmax(fit_curve)
            peak_positions.append(r[peak_idx])
        
        return np.std(peak_positions)
    except Exception as e:
        log.warning(f"Uncertainty calculation failed: {e}")
        return np.nan

def fit_first_peak(
    r: np.ndarray,
    g_r: np.ndarray,
    fit_method: Literal["gaussian", "skewed_gaussian", "emg", "none"] = "emg",
    smoothing_sigma: float = 2.0,
    min_threshold: float = 1.0,
    window_scale: float = 0.5
) -> Dict[str, Any]:
    """
    Find and fit the first peak in RDF data.
    
    Args:
        r: Distance array
        g_r: RDF values
        fit_method: Fitting method to use
        smoothing_sigma: Sigma for Gaussian smoothing
        min_threshold: Minimum g(r) value for peak detection
        window_scale: Window size in Ångströms around peak
        
    Returns:
        Dictionary containing results
    """
    # Smooth the data
    g_r_smooth = gaussian_filter1d(g_r, sigma=smoothing_sigma)
    
    # Initialize result dict
    result = {
        "r_peak": np.nan,
        "r_peak_uncertainty": np.nan,
        "fit_curve": np.zeros_like(r),
        "params": None,
        "covariance": None,
        "smoothed_rdf": g_r_smooth,
        "success": False
    }
    
    try:
        # Find peak window
        i_min, i_max = find_peak_window(r, g_r_smooth, min_threshold, window_scale)
        r_fit = r[i_min:i_max]
        g_fit = g_r_smooth[i_min:i_max]
        
        # Initial guess for parameters
        peak_idx = np.argmax(g_fit)
        r_peak_guess = r_fit[peak_idx]
        amp_guess = g_fit[peak_idx]
        sigma_guess = 0.2  # Reasonable starting guess for most RDFs
        
        if fit_method == "gaussian":
            p0 = [amp_guess, r_peak_guess, sigma_guess]
            bounds = ([0, min(r_fit), 0], [np.inf, max(r_fit), np.inf])
            popt, pcov = curve_fit(gaussian, r_fit, g_fit, p0=p0, bounds=bounds)
            fit_curve = gaussian(r, *popt)
            r_peak = popt[1]
            
        elif fit_method == "skewed_gaussian":
            p0 = [amp_guess, r_peak_guess, sigma_guess, 1.0]
            bounds = ([0, min(r_fit), 0, -np.inf], [np.inf, max(r_fit), np.inf, np.inf])
            popt, pcov = curve_fit(skewed_gaussian, r_fit, g_fit, p0=p0, bounds=bounds)
            fit_curve = skewed_gaussian(r, *popt)
            r_peak = r[np.argmax(fit_curve)]
            
        elif fit_method == "emg":
            p0 = [amp_guess, r_peak_guess, sigma_guess, 1.0, 1.0]
            bounds = ([0, min(r_fit), 0, 0, 0], [np.inf, max(r_fit), np.inf, np.inf, np.inf])
            popt, pcov = curve_fit(emg, r_fit, g_fit, p0=p0, bounds=bounds, maxfev=10000)
            fit_curve = emg(r, *popt)
            r_peak = popt[1]
            
        elif fit_method == "none":
            r_peak = r_peak_guess
            pcov = None
            popt = None
            fit_curve = np.zeros_like(r)
            
        else:
            raise ValueError(f"Unknown fit_method '{fit_method}'")
        
        # Calculate uncertainty
        if fit_method != "none":
            uncertainty = calculate_peak_uncertainty(r, eval(fit_method), popt, pcov)
        else:
            uncertainty = 0.0
        
        # Update results
        result.update({
            "r_peak": r_peak,
            "r_peak_uncertainty": uncertainty,
            "fit_curve": fit_curve,
            "params": popt,
            "covariance": pcov,
            "success": True
        })
        
    except Exception as e:
        log.warning(f"Peak fitting failed: {e}")
        # Fallback to simple peak detection
        peak_idx = np.argmax(g_r_smooth)
        result["r_peak"] = r[peak_idx]
        result["r_peak_uncertainty"] = np.nan
    
    return result

def plot_rdf(
    rdfs: defaultdict,
    save_path: Path,
    bin_width: float = 0.1,
    smoothing_sigma: float = 2.0,
    fit_method: Literal["gaussian", "skewed_gaussian", "emg", "none"] = "skewed_gaussian",
    min_threshold: float = 1.0,
    window_scale: float = 0.5
):
    """Plot RDFs with peak fitting."""
    n_rdfs = len(rdfs)
    n_cols = min(3, n_rdfs)
    n_rows = (n_rdfs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), squeeze=False)
    axes = axes.flatten()
    
    for ax, ((label_a, label_b), g_r_list) in zip(axes, rdfs.items()):
        try:
            g_r_array = np.stack(g_r_list)
            g_r_mean = np.mean(g_r_array, axis=0)
            r = 0.5 * (np.arange(len(g_r_mean)) + 0.5) * bin_width
            
            peak_fit = fit_first_peak(
                r, g_r_mean,
                fit_method=fit_method,
                smoothing_sigma=smoothing_sigma,
                min_threshold=min_threshold,
                window_scale=window_scale
            )
            
            ax.plot(r, g_r_mean, label="RDF (raw)", alpha=0.5)
            ax.plot(r, peak_fit["smoothed_rdf"], "--", label="RDF (smoothed)")
            
            if fit_method != "none" and peak_fit["success"]:
                ax.plot(r, peak_fit["fit_curve"], color="black", label=f"{fit_method} fit")
            
            ax.axvline(
                peak_fit["r_peak"],
                color="red",
                linestyle=":",
                label=f"$r_\\mathrm{{peak}}$ = {peak_fit['r_peak']:.2f} ± {peak_fit['r_peak_uncertainty']:.2f} Å"
            )
            
            ax.set_xlabel("Distance r (Å)")
            ax.set_ylabel("g(r)")
            ax.set_title(f"RDF: {label_a} - {label_b}")
            ax.legend()
            ax.grid(True)
            
        except Exception as e:
            log.error(f"Failed to plot {label_a}-{label_b}: {e}")
            ax.set_title(f"Error plotting {label_a}-{label_b}")
    
    # Hide unused axes
    for ax in axes[len(rdfs):]:
        ax.set_visible(False)
    
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)