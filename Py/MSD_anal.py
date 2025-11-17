import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

def calculate_diffusion_exponent(file_path, dt, output_interval, fit_start_time=100.0):
    """
    Analyzes an MSD data file to calculate the diffusion exponent (alpha).
    Returns the exponent as a float, or NaN if fitting fails.
    """
    try:
        msd_data = np.loadtxt(file_path)
        if msd_data.size < 20: return np.nan

        time_per_point = dt * output_interval
        time = np.arange(1, len(msd_data) + 1) * time_per_point

        fit_mask = time >= fit_start_time
        if np.sum(fit_mask) < 5: return np.nan
        
        valid_log_mask = (msd_data[fit_mask] > 0)
        log_time = np.log(time[fit_mask][valid_log_mask])
        log_msd = np.log(msd_data[fit_mask][valid_log_mask])
        
        if len(log_time) < 2: return np.nan

        alpha, _ = np.polyfit(log_time, log_msd, 1)
        return alpha
        
    except Exception:
        return np.nan

# --- MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    # --- Simulation Parameters ---
    SIM_DT = 0.01
    SIM_OUTPUT_INTERVAL = 1000
    DATA_DIR = '../MSD/' 
    PLOT_DIR = '../Plot/'

    # 1. Scan directory and analyze all MSD files
    analysis_results = []
    filename_pattern = re.compile(r'A2_(\d+\.?\d*)_tau_(\d+\.?\d*)_beta_(\d+\.?\d*)\.csv$')

    print("Scanning for MSD files...")
    for filename in os.listdir(DATA_DIR):
        match = filename_pattern.match(filename)
        if match:
            try:
                a2, tau, beta = [float(v) for v in match.groups()]
                full_path = os.path.join(DATA_DIR, filename)
                alpha = calculate_diffusion_exponent(full_path, SIM_DT, SIM_OUTPUT_INTERVAL)
                
                if not np.isnan(alpha):
                    print(f"  -> Analyzed: A2={a2:.2f}, tau={tau:.2f}, beta={beta:.2f} -> alpha = {alpha:.3f}")
                    analysis_results.append({'A2': a2, 'tau': tau, 'beta': beta, 'alpha': alpha})
            except Exception as e:
                print(f"Could not process file {filename}: {e}")

    if not analysis_results:
        print("\nNo data files were found to analyze.")
    else:
        # 2. Create a pandas DataFrame for plotting
        df = pd.DataFrame(analysis_results)
        
        # Create a new column for clear labels in the plot legend
        df['Case'] = 'τ=' + df['tau'].astype(str) + ', β=' + df['beta'].astype(str)
        df = df[df.A2 != 0.5]
        print("\n--- Analysis Summary ---")
        print(df.to_string())
        
        # 3. Create a single figure for all curves
        plt.figure(figsize=(12, 8))
        
        # Use seaborn to automatically plot the different curves on one figure
        # 'hue' will set the color, and 'style' will set the line/marker style
        sns.lineplot(data=df, x='A2', y='alpha', hue='Case', style='Case',
                     markers=True, dashes=True, markersize=8)

        plt.title(r'Diffusion Exponent ($\alpha$) vs. Amplitude ($A_2$)', fontsize=18)
        plt.xlabel('$A_2$', fontsize=16)
        plt.ylabel(r'Diffusion Exponent, $\alpha$', fontsize=16)
        plt.grid(True, which="both", ls="--")
        plt.legend(title='Parameters')
        plt.tight_layout()

        os.makedirs(PLOT_DIR, exist_ok=True)
        save_path = os.path.join(PLOT_DIR, "summary_plot_all_cases.png")
        plt.savefig(save_path, dpi=300)
        print(f"\nCombined summary plot saved to: {save_path}")