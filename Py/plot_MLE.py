import matplotlib.pyplot as plt
import numpy as np
import os 
import re

plt.rcParams.update({
"text.usetex": True,
"font.family": "serif", # Use Times New Roman or similar
"font.size": 10, # Base font size
"axes.labelsize": 12, # Axis labels
"xtick.labelsize": 10, # X-ticks
"ytick.labelsize": 10, # Y-ticks
"figure.dpi": 300, # High resolution
"figure.autolayout": False # Disable auto-layout (use constrained_layout instead)
})

# --- NEW Plotting Function ---
# This function is adapted to handle three parameters for the title and filename.
def plot_strobo_3_params(data, A2, tau, beta, xlabel=r'$x$', ylabel=r'$y$'):
    """
    Plots the given data and saves the figure with a filename containing A2, tau, and beta.
    """
    fig, ax = plt.subplots()
    data_clipped = np.clip(data, 0, None)
    im = ax.imshow(data_clipped, aspect='auto', extent=(0, np.pi, 0, 2 * np.pi),origin='lower',
                   cmap='viridis_r',
                   interpolation='nearest')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r'$\lambda$', fontsize=12)
    # The title now includes all three parameters for better context
    ax.set_title(f'$\\tau = {tau:.2f}, \\nu = {beta:.2f}$')
    ax.set_xlim(0, np.pi)
    ax.set_ylim(0, 2 * np.pi)
    
    # Ensure the output directory exists
    output_dir = '../Plot'
    os.makedirs(output_dir, exist_ok=True)
    
    # The saved filename now includes all three parameters
    save_path = f'{output_dir}/MLE_A2_{A2:.4f}_tau_{tau:.4f}_beta_{beta:.4f}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot: {save_path}")


def natural_sort_key(filename):
    """A key function for natural sorting of strings containing numbers."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]


# --- NEW Data Loading Function ---
# This function is adapted to parse filenames with three numerical values.
def load_csv_files_with_three_values(directory, filter_func=None):
    """
    Loads CSV files from a directory that match "A<..._value1_value2_value3.csv".
    It extracts the three values and can optionally filter files.
    """
    f_data = []
    extracted_triplets = []

    # --- UPDATED REGEX ---
    # This regex now looks for three groups of numbers after the initial "A2_" part.
    # It assumes a filename format like: A2_0.10_0.50_1.57.csv
    filename_pattern = re.compile(r'A\d+_(\d+\.?\d*)_tau_(\d+\.?\d*)_beta_(\d+\.?\d*)\.csv$')

    try:
        all_items_in_directory = os.listdir(directory)
    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.")
        return [], []

    csv_basenames = sorted(
        [f for f in all_items_in_directory if f.endswith('.csv') and os.path.isfile(os.path.join(directory, f))],
        key=natural_sort_key
    )

    for basename in csv_basenames:
        match = filename_pattern.match(basename)
        
        if match:
            # --- UPDATED UNPACKING ---
            # Now captures three values from the filename
            val1_str, val2_str, val3_str = match.groups()
            
            try:
                val1 = float(val1_str) # A2
                val2 = float(val2_str) # tau
                val3 = float(val3_str) # beta
                
                # --- UPDATED FILTER ---
                # The filter function (if provided) should now accept three arguments
                if filter_func is None or filter_func(val1, val2, val3):
                    full_file_path = os.path.join(directory, basename)
                    try:
                        # Your C++ code used tabs, so we specify delimiter='\t'
                        data_array = np.loadtxt(full_file_path)
                        print(f"Loaded data from '{full_file_path}' with shape {data_array.shape}")
                        # Check if the file is empty or malformed
                        #if data_array.size == 0:
                        #    print(f"Warning: File '{full_file_path}' is empty. Skipping.")
                        #    continue

                        f_data.append(data_array)
                        extracted_triplets.append([val1, val2, val3])
                    except Exception as load_err:
                        print(f"Warning: Error loading '{full_file_path}': {load_err}. Skipping.")
                
            except ValueError:
                print(f"Warning: Could not convert values to float for '{basename}'. Skipping.")
            
    return f_data, extracted_triplets


# --- MAIN EXECUTION BLOCK ---

# Set the directory where your C++ output files are stored
csv_directory = "../LE"

# --- UPDATED FUNCTION CALL ---
# Call the new function designed for three parameters
# We pass no filter_func, so it will load all matching files.
vetor, name_parameter_triplets = load_csv_files_with_three_values(csv_directory)

if not vetor:
    print("No matching data files were found or loaded. Please check the directory and filenames.")
else:
    print(f"Successfully loaded data for {len(vetor)} files. Starting to plot...")

# Loop through the loaded data and plot each one

for i, data in enumerate(vetor):
   
        
        
        # --- UPDATED PARAMETER UNPACKING ---
        # Get the corresponding [A2, tau, beta] triplet
        current_params = name_parameter_triplets[i]
        A2, tau, beta = current_params
        
        # Call the new plotting function with all three parameters
        if A2 == 0.1:
            plot_strobo_3_params(data, A2, tau, beta, xlabel='x', ylabel='y')
        


print("Processing complete.")