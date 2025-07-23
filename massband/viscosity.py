import matplotlib.pyplot as plt
import numpy as np
import pint
import znh5md
import zntrack
from pathlib import Path

ureg = pint.UnitRegistry()


class ViscosityCalculator(zntrack.Node):
    """Calculate viscosity from stress tensor autocorrelation using Green-Kubo relations.
    
    The viscosity is computed using:
    η = (V / k_B T) ∫₀^∞ ⟨σ_αβ(t)σ_αβ(0)⟩ dt
    
    where σ_αβ are the off-diagonal stress tensor components.
    
    References
    ----------
    Allen, M. P., & Tildesley, D. J. (2017). Computer simulation of liquids. 
    Oxford university press.
    """
    
    filename: str = zntrack.deps_path()
    temperature: float = zntrack.params()
    timestep: float = zntrack.params()  # in femtoseconds
    sampling_rate: int = zntrack.params(1)  # every N frames
    metrics: dict = zntrack.metrics()
    figures: Path = zntrack.outs_path(zntrack.nwd / "figures")
    
    def run(self):
        io = znh5md.IO(filename=self.filename, include=["position", "stress", "box"])
        
        # Extract all stress tensors
        stress_tensors = [x.get_stress() for x in io[:]]
        stress_tensors = np.array(stress_tensors)  # Shape: (n_frames, 6) - Voigt notation
        
        # Get volume from first frame
        volume = io[0].get_volume() * ureg.angstrom**3
        
        # Extract off-diagonal stress components (xy, xz, yz) from Voigt notation
        # Voigt: [xx, yy, zz, xy, xz, yz] -> indices [3, 4, 5]
        stress_xy = stress_tensors[:, 3] * ureg.eV / ureg.angstrom**3
        stress_xz = stress_tensors[:, 4] * ureg.eV / ureg.angstrom**3
        stress_yz = stress_tensors[:, 5] * ureg.eV / ureg.angstrom**3
        
        # Calculate autocorrelation functions for each shear component
        n_frames = len(stress_tensors)
        max_lag = n_frames // 2  # Use half the trajectory for good statistics
        
        def autocorr(data):
            """Calculate autocorrelation function using FFT for efficiency."""
            n = len(data)
            # Zero-pad the data
            padded = np.zeros(2 * n)
            padded[:n] = data - np.mean(data)
            
            # FFT-based autocorrelation
            fft = np.fft.fft(padded)
            autocorr_full = np.fft.ifft(fft * np.conj(fft)).real
            
            # Normalize and return first half
            autocorr_norm = autocorr_full[:n] / autocorr_full[0]
            return autocorr_norm
        
        # Calculate autocorrelations for each shear component
        # Extract magnitude values from pint quantities
        stress_xy_vals = stress_xy.magnitude
        stress_xz_vals = stress_xz.magnitude
        stress_yz_vals = stress_yz.magnitude
        
        acf_xy = autocorr(stress_xy_vals)
        acf_xz = autocorr(stress_xz_vals)
        acf_yz = autocorr(stress_yz_vals)
        
        # Average the three shear autocorrelations
        acf_avg = (acf_xy + acf_xz + acf_yz) / 3.0
        
        # Calculate effective time step from timestep and sampling rate
        dt = self.timestep * self.sampling_rate * ureg.femtosecond
        
        # Integrate autocorrelation function using trapezoidal rule
        # Only integrate up to where the function decays significantly
        cutoff_idx = self._find_integration_cutoff(acf_avg[:max_lag])
        dt_val = float(dt.magnitude)
        integral = np.trapezoid(acf_avg[:cutoff_idx], dx=dt_val)
        
        # Calculate viscosity using Green-Kubo relation
        prefactor = volume / (ureg.boltzmann_constant * self.temperature * ureg.kelvin)
        # Units: volume/(kB*T) * integral * stress_units^2
        stress_unit_squared = (ureg.eV / ureg.angstrom**3)**2
        viscosity_quantity = prefactor * integral * ureg.femtosecond * stress_unit_squared
        viscosity = viscosity_quantity.to(ureg.Pa * ureg.s)
        
        # Store results
        self.metrics = {
            "viscosity": viscosity.magnitude,
            "viscosity_units": str(viscosity.units),
            "integration_cutoff_frames": cutoff_idx,
            "max_autocorr_lag": max_lag,
            "n_frames_used": n_frames,
            "temperature": self.temperature,
            "volume": float(volume.to(ureg.nm**3).magnitude)
        }
        
        # Create autocorrelation plots
        self.figures.mkdir(exist_ok=True, parents=True)
        
        # Only plot up to max_lag for better visualization
        plot_length = min(max_lag, len(acf_avg))
        times = np.arange(plot_length) * dt_val  # in femtoseconds
        
        # First plot: Full autocorrelation functions
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Convert times to picoseconds for plotting
        times_ps = times / 1000.0
        cutoff_time_ps = (cutoff_idx * dt_val) / 1000.0
        
        # Plot individual autocorrelations (truncated to plot_length)
        ax1.plot(times_ps, acf_xy[:plot_length], label='σ_xy', alpha=0.7)
        ax1.plot(times_ps, acf_xz[:plot_length], label='σ_xz', alpha=0.7)
        ax1.plot(times_ps, acf_yz[:plot_length], label='σ_yz', alpha=0.7)
        ax1.plot(times_ps, acf_avg[:plot_length], 'k-', linewidth=2, label='Average')
        ax1.axvline(cutoff_time_ps, color='red', linestyle='--', label=f'Cutoff ({cutoff_idx} frames)')
        ax1.set_xlabel('Time t / ps')
        ax1.set_ylabel('Stress Autocorrelation')
        ax1.set_title('Stress Tensor Autocorrelation Functions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot zoomed view around cutoff
        zoom_end = min(cutoff_idx * 3, plot_length)
        times_zoom_ps = times_ps[:zoom_end]
        ax2.plot(times_zoom_ps, acf_avg[:zoom_end], 'k-', linewidth=2)
        ax2.axvline(cutoff_time_ps, color='red', linestyle='--', label='Integration cutoff')
        ax2.axhline(0, color='gray', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Time t / ps')
        ax2.set_ylabel('Average Stress Autocorrelation')
        ax2.set_title('Integration Region (Zoomed)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures / 'stress_autocorrelation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Second plot: Enhanced view up to cutoff with fixed y-axis bounds
        # (cutoff_time_ps already calculated above)
        
        fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Plot up to cutoff with enhanced scaling
        cutoff_range = slice(0, cutoff_idx + 1)
        time_cutoff_ps = times[cutoff_range] / 1000.0  # Convert fs to ps
        
        ax.plot(time_cutoff_ps, acf_xy[cutoff_range], label='σ_xy', alpha=0.8, linewidth=1.5)
        ax.plot(time_cutoff_ps, acf_xz[cutoff_range], label='σ_xz', alpha=0.8, linewidth=1.5)
        ax.plot(time_cutoff_ps, acf_yz[cutoff_range], label='σ_yz', alpha=0.8, linewidth=1.5)
        ax.plot(time_cutoff_ps, acf_avg[cutoff_range], 'k-', linewidth=2.5, label='Average')
        
        ax.axvline(cutoff_time_ps, color='red', linestyle='--', linewidth=2, label='Integration cutoff')
        ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
        
        # Set fixed y-limits and x-limits
        ax.set_ylim(-0.1, 0.1)
        ax.set_xlim(0, cutoff_time_ps * 1.1)
        
        ax.set_xlabel('Time t / ps')
        ax.set_ylabel('Stress Autocorrelation')
        ax.set_title(f'Integration Region Detail (up to {cutoff_time_ps:.2f} ps)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures / 'stress_autocorrelation_detail.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Viscosity: {viscosity:.3e}")
        print(f"Integration cutoff at frame: {cutoff_idx} ({cutoff_idx * dt_val:.1f} fs)")
        print(f"Used {n_frames} frames from trajectory")
        print(f"Effective timestep: {dt_val:.1f} fs (timestep={self.timestep} fs × sampling_rate={self.sampling_rate})")
        print(f"Autocorrelation plots saved to: {self.figures}")
        print(f"  - Full view: stress_autocorrelation.png")
        print(f"  - Detail view: stress_autocorrelation_detail.png")
    
    def _find_integration_cutoff(self, acf, threshold=0.05):
        """Find where autocorrelation function decays below threshold."""
        # Find first point where ACF drops below threshold
        below_threshold = np.where(np.abs(acf) < threshold)[0]
        if len(below_threshold) > 0:
            # Use first zero crossing or threshold crossing, whichever comes first
            zero_crossings = np.where(np.diff(np.sign(acf)))[0]
            if len(zero_crossings) > 0:
                cutoff = min(below_threshold[0], zero_crossings[0])
            else:
                cutoff = below_threshold[0]
        else:
            # If never drops below threshold, use 1/4 of the data
            cutoff = len(acf) // 4
        
        # Ensure we use at least 100 points for reasonable statistics
        return max(cutoff, 100)