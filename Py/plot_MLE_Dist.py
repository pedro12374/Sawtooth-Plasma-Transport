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
    "figure.figsize": [8, 4],     # Figure size
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

fig, ax = plt.subplots()

g = sns.catplot(
    data=df_long,
    x='tau',
    y='data',
    col='beta',          # Creates subplots based on the 'tau' column
    kind='violin',
    col_wrap=2,         # Wraps the subplots into a 2xN grid
    
    # 2. Fix the palette warning by explicitly assigning hue
    hue='tau',         # Map hue to the same variable as x
    palette='muted',
    legend=False        # Hide the legend as it's redundant with the x-axis
)

g.set_titles(r"$\nu$ = {col_name}")

# 4. (Optional) Customize the main axis labels
g.set_axis_labels(r"$\tau$", r"$\lambda$")

plt.savefig('../Plot/LE_Dist.png', bbox_inches='tight', dpi=300)