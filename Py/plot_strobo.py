import matplotlib.pyplot as plt
import numpy as np
import os 
import re

# --- NEW Plotting Function ---
# This function is adapted to handle three parameters for the title and filename.
def plot_strobo_3_params(t, x, y, A2, tau, beta, xlabel='x', ylabel='y'):
    """
    Plots the given data and saves the figure with a filename containing A2, tau, and beta.
    """
    fig, ax = plt.subplots()
    ax.plot(x, y, 'k,')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # The title now includes all three parameters for better context
    ax.set_title(f'$A_2 = {A2:.2f}, \\tau = {tau:.2f}, \\beta = {beta:.2f}$')
    ax.set_xlim(0, np.pi)
    ax.set_ylim(0, 2 * np.pi)
    
    # Ensure the output directory exists
    output_dir = '../Plot'
    os.makedirs(output_dir, exist_ok=True)
    
    # The saved filename now includes all three parameters
    save_path = f'{output_dir}/strobo_A2_{A2:.4f}_tau_{tau:.4f}_beta_{beta:.4f}.png'
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
    filename_pattern = re.compile(r'A\d+_(\d+\.?\d*)_(\d+\.?\d*)_(\d+\.?\d*)\.csv$')

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
                        data_array = np.loadtxt(full_file_path, unpack=True, delimiter='\t')
                        
                        # Check if the file is empty or malformed
                        if data_array.size == 0:
                            print(f"Warning: File '{full_file_path}' is empty. Skipping.")
                            continue

                        f_data.append(data_array)
                        extracted_triplets.append([val1, val2, val3])
                    except Exception as load_err:
                        print(f"Warning: Error loading '{full_file_path}': {load_err}. Skipping.")
                
            except ValueError:
                print(f"Warning: Could not convert values to float for '{basename}'. Skipping.")
            
    return f_data, extracted_triplets


# --- MAIN EXECUTION BLOCK ---

# Set the directory where your C++ output files are stored
csv_directory = "../Strobo"

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
    try:
        # Unpack the data columns
        t, x, y = data
        
        # --- UPDATED PARAMETER UNPACKING ---
        # Get the corresponding [A2, tau, beta] triplet
        current_params = name_parameter_triplets[i]
        A2, tau, beta = current_params
        
        # Call the new plotting function with all three parameters
        plot_strobo_3_params(t, x, y, A2, tau, beta, xlabel='x', ylabel='y')
        
    except ValueError:
        print(f"Warning: Could not unpack data for file index {i}. It might have an incorrect number of columns. Skipping.")
    except Exception as e:
        print(f"An unexpected error occurred while processing file index {i}: {e}")

print("Processing complete.")