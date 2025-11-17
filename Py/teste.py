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
                #'a2_param': float(a2_val),
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

Tot_directory = "../Displ"
x_directory = "../Displ_x"
y_directory = "../Displ_y"

df_D = load_data_to_dataframe(Tot_directory)

df_long_filtered = df_D.explode('data')

df_long_filtered = df_long_filtered[df_long_filtered['data'] <= 40]
df_long_filtered = df_long_filtered.sample(frac=0.01, random_state=1)
g = sns.displot(
    data=df_long_filtered,
    x="data",
    hue="tau",
    col="beta",
    kind="kde",
    col_wrap=3,       # Wrap the panels into a grid with 3 columns
    palette="viridis",  # Use a nice color palette
    height=4,
    aspect=1.2
)

g.tight_layout()

plt.savefig("kde_panels_plot.png")
plt.close()