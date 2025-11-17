import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os 
import re
from scipy.optimize import curve_fit
import parana_theme as tema
tema.aplicar_tema()
# Set the seaborn theme
sns.set_theme(
    context='paper',     # 'paper', 'notebook', 'talk', 'poster'
    style='ticks',       # 'darkgrid', 'whitegrid', 'dark', 'white', 'ticks'
)




def function(x,a,b):
    return a*x**b

def calculate_diffusion_exponent(file_path, dt, output_interval):
    """
    Analyzes an MSD data file to calculate the diffusion exponent (alpha).
    Returns the exponent as a float, or NaN if fitting fails.
    """

    msd_data = np.loadtxt(file_path)


    time_per_point = dt * output_interval
    time = np.arange(0, output_interval+dt,dt)
    popt, pcov = curve_fit(function,time, msd_data)
    #print(time)
    return popt[1]

analysis_results = []
filename_pattern = re.compile(r'A2_0.1000_tau_(\d+\.?\d*)_beta_(\d+\.?\d*)\.csv$')


SIM_DT = 0.01
SIM_OUTPUT_INTERVAL = 1000
DATA_DIR = '../MSD/' 
PLOT_DIR = '../tese/'

#print("Scanning for MSD files...")
for filename in os.listdir(DATA_DIR):
    match = filename_pattern.match(filename)
    if match:
        try:
            tau, beta = [float(v) for v in match.groups()]
            full_path = os.path.join(DATA_DIR, filename)
            
            alpha = calculate_diffusion_exponent(full_path, SIM_DT, SIM_OUTPUT_INTERVAL)
            
            if not np.isnan(alpha):
                #print(f"  -> Analyzed: tau={tau:.2f}, beta={beta:.2f} -> alpha = {alpha:.3f}")
                analysis_results.append({ 'tau': tau, 'beta': beta, 'alpha': alpha})
        except Exception as e:
            print(f"Could not process file {filename}: {e}")

if not analysis_results:
    print("\nNo data files were found to analyze.")
else:
    # 2. Create a pandas DataFrame for plotting
    df = pd.DataFrame(analysis_results)
    
    # Create a new column for clear labels in the plot legend
    df['Case'] = 'τ=' + df['tau'].astype(str) + ', β=' + df['beta'].astype(str)


fig, ax = plt.subplots()
fig.set_size_inches(8,3)
sns.lineplot(data=df, x='tau', y='alpha', hue='beta', marker='o',palette=tema.parana_jet ,ax=ax)
ax.set_xlabel(r'$\tau$')
ax.set_ylabel(r'$\alpha$')
ax.legend(title=r'$\nu$')

plt.savefig(os.path.join(PLOT_DIR, 'Serra-MSD.pdf'), bbox_inches='tight', format='pdf')