import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os 
import re
from scipy.optimize import curve_fit
from scipy import stats

# Set the seaborn theme
sns.set_theme(
    context='paper',     # 'paper', 'notebook', 'talk', 'poster'
    style='ticks',       # 'darkgrid', 'whitegrid', 'dark', 'white', 'ticks'
)


plt.rcParams.update({
    "figure.figsize": [5, 4],     # Figure size
     "text.usetex": True, 
    "font.family": "serif",       # Use Times New Roman or similar
    "font.size": 12,              # Base font size
    "axes.labelsize": 14,         # Axis labels
    "xtick.labelsize": 12,        # X-ticks
    "ytick.labelsize": 12,        # Y-ticks
    "legend.fontsize": 11,         # Legend
    "figure.dpi": 600,            # High resolution for raster exports
})

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
filename_pattern = re.compile(r'A2_(\d+\.?\d*)_tau_(\d+\.?\d*)_beta_(\d+\.?\d*)\.csv$')


SIM_DT = 0.01
SIM_OUTPUT_INTERVAL = 1000
DATA_DIR = '../MSD/' 
PLOT_DIR = '../Plot/'

#print("Scanning for MSD files...")
for filename in os.listdir(DATA_DIR):
    match = filename_pattern.match(filename)
    if match:
        try:
            A2, tau, beta = [float(v) for v in match.groups()]
            full_path = os.path.join(DATA_DIR, filename)
            
            alpha = calculate_diffusion_exponent(full_path, SIM_DT, SIM_OUTPUT_INTERVAL)
            
            if not np.isnan(alpha):
                #print(f"  -> Analyzed: tau={tau:.2f}, beta={beta:.2f} -> alpha = {alpha:.3f}")
                analysis_results.append({ 'A2': A2, 'tau': tau, 'beta': beta, 'alpha': alpha})
        except Exception as e:
            print(f"Could not process file {filename}: {e}")

if not analysis_results:
    print("\nNo data files were found to analyze.")
else:
    # 2. Create a pandas DataFrame for plotting
    df_alpha = pd.DataFrame(analysis_results)
    
    # Create a new column for clear labels in the plot legend
    df_alpha['Case'] = 'τ=' + df_alpha['tau'].astype(str) + ', β=' + df_alpha['beta'].astype(str)

def load_data_to_dataframe(directory):
    """
    Scans a directory for CSV files containing matrix data, extracts 
    parameters from filenames, loads each matrix, flattens it, and 
    returns everything in a single pandas DataFrame.

    Args:
        directory (str): The path to the directory containing the CSV files.

    Returns:
        pandas.DataFrame: A DataFrame where each row corresponds to a file,
                          with columns for parameters and the flattened data.
    """
    pattern = re.compile(r'A2_(\d+\.?\d*)_tau_(\d+\.?\d*)_beta_(\d+\.?\d*)\.csv$')
    
    all_files_data = []

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        
        if match:
            A2_val, tau_val, beta_val = match.groups()
            file_path = os.path.join(directory, filename)
            
            # --- This is the key change ---
            # 1. Read the full MxM matrix into a DataFrame
            matrix_df = pd.read_csv(file_path, header=None,delim_whitespace=True)
            
            # 2. Get the underlying NumPy array and flatten it into a 1D array
            flattened_data = matrix_df.values.flatten()
            # -----------------------------
            
            data_row = {
                'A2': float(A2_val),
                'tau': float(tau_val),
                'beta': float(beta_val),
                # The 'data' column now holds a 1D NumPy array
                'data': flattened_data
            }
            
            all_files_data.append(data_row)
            
    if not all_files_data:
        print("⚠️ No matching files were found.")
        return pd.DataFrame()
        
    return pd.DataFrame(all_files_data)

Dir = '../LE/'
LE = load_data_to_dataframe(Dir)
df_long = LE.explode('data')

# 2. (Optional but good practice) Ensure the data column is a numeric type
df_long['data'] = pd.to_numeric(df_long['data'])


df_summary = df_long[['A2','tau', 'beta']].copy()
df_summary = df_long.groupby(['A2','tau', 'beta'])['data'].mean().reset_index(name='mean_value')

merged_df = pd.merge(
    df_summary, 
    df_alpha, 
    on=['A2','tau', 'beta'], 
    how='inner'
)


fig, ax = plt.subplots()
#sns.regplot(data=merged_df, x='mean_value', y='alpha',ax=ax,scatter_kws={"s": 50, "alpha": 0.7, "color": "teal"}, line_kws={"color": "orange"})



pearson = merged_df['mean_value'].corr(merged_df['alpha'], method='pearson')
spearman = merged_df['mean_value'].corr(merged_df['alpha'], method='spearman')



def exponential_decay(x, A, k):
    """Exponential decay model with a baseline of 1.0."""
    return A * np.exp(-k * x) + 1.0

# 2. Extract your data
x_data = merged_df['mean_value']
y_data = merged_df['alpha']

# 3. Perform the non-linear fit
#    popt will contain the optimized values for [A, k]
popt, pcov = curve_fit(exponential_decay, x_data, y_data)
A_fit, k_fit = popt
print(f"Fit Parameters: A = {A_fit:.3f}, k = {k_fit:.3f}")

perr = np.sqrt(np.diag(pcov))

# Create a function to calculate the derivatives of the model with respect to its parameters
def jacobian(x, A, k):
    return np.array([np.exp(-k * x), -A * x * np.exp(-k * x)]).T

confidence_level = 0.95
n_params = len(popt)
dof = max(0, len(x_data) - n_params) # degrees of freedom
t_value = np.abs(stats.t.ppf((1 - confidence_level) / 2., dof))

x_smooth = np.linspace(x_data.min(), x_data.max(), 200)
y_fit = exponential_decay(x_smooth, A_fit, k_fit)

J = jacobian(x_smooth, A_fit, k_fit)
y_cov = J @ pcov @ J.T
y_std_err = np.sqrt(np.diag(y_cov))

upper_bound = y_fit + t_value * y_std_err
lower_bound = y_fit - t_value * y_std_err


fit_label = (
    fr'Pearson r: {pearson:.2f}\\'
    fr'Spearman $\rho$: {spearman:.2f}'
)


# Plot the original data points
ax.scatter(x_data, y_data, label='Original Data', color='teal', s=50, alpha=0.7)

# Generate a smooth line for the fitted curve
x_smooth = np.linspace(x_data.min(), x_data.max(), 200)
y_fit = exponential_decay(x_smooth, A_fit, k_fit)

# Plot the fitted exponential curve
ax.plot(x_smooth, y_fit, color='orange', linewidth=2.0,zorder=3)

# Plot the confidence interval as a shaded area
ax.fill_between(x_smooth, lower_bound, upper_bound, color='orange', alpha=0.2, label=f'{int(confidence_level*100)}% Confidence Interval', zorder=1)

ax.text(0.6, 0.95, fit_label, transform=ax.transAxes)




ax.set_xlabel(r'$\Lambda$')
ax.set_ylabel(r'$\alpha$')

plt.savefig('../Plot/Regplot2.png', bbox_inches='tight', dpi=300)