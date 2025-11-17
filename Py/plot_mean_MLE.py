import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os 
import re
from math import hypot
from scipy.stats import norm
from scipy import stats
from matplotlib.transforms import ScaledTranslation

# Set the seaborn theme
sns.set_theme(
    context='paper',     # 'paper', 'notebook', 'talk', 'poster'
    style='ticks',       # 'darkgrid', 'whitegrid', 'dark', 'white', 'ticks'
)


plt.rcParams.update({
    "figure.figsize": [12, 6],     # Figure size
     "text.usetex": True, 
    "font.family": "serif",       # Use Times New Roman or similar
    "font.size": 12,              # Base font size
    "axes.labelsize": 14,         # Axis labels
    "xtick.labelsize": 12,        # X-ticks
    "ytick.labelsize": 12,        # Y-ticks
    "legend.fontsize": 11,         # Legend
    "figure.dpi": 600,            # High resolution for raster exports
})

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
    pattern = re.compile(r'A2_0.1000_tau_(\d+\.?\d*)_beta_(\d+\.?\d*)\.csv$')
    
    all_files_data = []

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        
        if match:
            tau_val, beta_val = match.groups()
            file_path = os.path.join(directory, filename)
            
            # --- This is the key change ---
            # 1. Read the full MxM matrix into a DataFrame
            matrix_df = pd.read_csv(file_path, header=None,delim_whitespace=True)
            
            # 2. Get the underlying NumPy array and flatten it into a 1D array
            flattened_data = matrix_df.values.flatten()
            # -----------------------------
            
            data_row = {
                
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
df_long = df_long[df_long['beta'] != 3.0]
desired_taus = np.round(np.arange(0.1, 1.01, 0.2), 2)

df_long = df_long[df_long['tau'].isin(desired_taus)].copy()

df_summary = df_long[['tau', 'beta']].copy()
df_summary = df_long.groupby(['tau', 'beta'])['data'].mean().reset_index(name='mean_value')




name = [["a)","b)"]]


fig, axs = plt.subplot_mosaic(name,layout='constrained', gridspec_kw={
        "wspace": -0.1,
        "hspace": -0.1,
    },)


# Box plot for Tau vs. Mean Value
sns.boxplot(ax=axs["a)"], data=df_summary, x='tau', y='mean_value',color="peachpuff")

axs["a)"].set_xlabel(r'$\tau$')
axs["a)"].set_ylabel(r'$\Lambda$')

# Box plot for Beta vs. Mean Value
sns.boxplot(ax=axs["b)"], data=df_summary, x='beta', y='mean_value',color="lemonchiffon")

axs["b)"].set_xlabel(r'$\beta$')
axs["b)"].set_ylabel('') # Hide y-label for the second plot for a cleaner look


for label, ax in axs.items():
    # Use ScaledTranslation to put the label
    # - at the top left corner (axes fraction (0, 1)),
    # - offset 20 pixels left and 7 pixels up (offset points (-20, +7)),
    # i.e. just outside the axes.
    ax.text(
        0.0, 1.0, label, transform=(
            ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
        fontsize='large', va='bottom', fontfamily='serif')


plt.savefig('../Plot/LE_BoxPlot.png', bbox_inches='tight', dpi=300)