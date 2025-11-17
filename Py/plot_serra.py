import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",       # Use Times New Roman or similar
    "font.size": 12,              # Base font size
    "axes.labelsize": 14,         # Axis labels
    "xtick.labelsize": 12,        # X-ticks
    "ytick.labelsize": 12,        # Y-ticks
    "figure.dpi": 600,            # High resolution
    "figure.autolayout": False    # Disable auto-layout (use constrained_layout instead)
})



# --- You can now change these values dynamically ---
tau = 0.1   # Period
nu = 5.0  # Parameter for pulse width
# --- T is now set equal to tau automatically ---
T = tau
A_pulse = 0.35
# Generate a time array to show a few cycles
# The time range adjusts based on tau
t = np.linspace(-1, 4 * tau, 1000)

# --- Equation 1: Rectangular Pulse Train ---
pulse_width = tau / nu
t_mod_tau = np.fmod(t, tau)
# The expression below is 1 when 0 <= t_mod_tau < pulse_width, and 0 otherwise.
rectangular_pulse = A_pulse * (np.heaviside(t_mod_tau, 1) - np.heaviside(t_mod_tau - pulse_width, 1))

# --- 2. Calculate the Sawtooth Wave ---
# This creates a wave that ramps from 0 to 1 over each period tau.
sawtooth_wave = (t / tau) - np.floor(t / tau)

# --- Plotting a single combined plot ---
fig, ax = plt.subplots(figsize=(12, 4))

# Plot both waveforms
ax.plot(t, rectangular_pulse, label='Rectangular Pulse', linewidth=2)
ax.plot(t, sawtooth_wave, label='Sawtooth Wave', linestyle='--', linewidth=2)

# --- DYNAMIC ANNOTATIONS ---

# 1. Annotate Tau (Period)
# The arrow is now placed from t=tau to t=2*tau
ax.annotate('', xy=(tau, -0.15), xytext=(2 * tau, -0.15),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax.text(1.5 * tau, -0.2, r'$\tau$ (Period)', ha='center', va='top', fontsize=14)

# Annotate the Pulse Width (τ/ν)
# This arrow highlights the pulse in the second period for clarity
pulse_start_time = tau
pulse_end_time = tau + pulse_width
ax.annotate('', xy=(pulse_start_time, 0.45), xytext=(pulse_end_time, 0.45),
            arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
ax.text(pulse_start_time + pulse_width / 2, 0.5, r'Pulse Width = $\tau/\nu$',
        ha='center', va='bottom', fontsize=14, color='blue')

# --- Final plot settings ---

ax.set_xlabel('Time (t)')
ax.set_ylabel('Amplitude')
ax.grid(True)
ax.set_ylim(-0.3, 1.2)
ax.set_xlim(0, 3 * tau)
ax.legend()

plt.savefig('../Plot/Serra_A2_tau_beta.png', dpi=300, bbox_inches='tight')