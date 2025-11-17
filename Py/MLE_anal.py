import numpy as np
import matplotlib.pyplot as plt
import os
import re
from collections import defaultdict

def natural_sort_key(filename):
    """A key function for natural sorting of strings containing numbers."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

def plot_grouped_distributions(a2_value, case_list, bins=100, output_dir='.'):
    """
    Plots multiple LE distributions on a single figure for a given A2 value.

    Args:
        a2_value (float): The A2 value for this group of plots.
        case_list (list): A list of dictionaries, where each dict contains data and params.
        bins (int, optional): The number of bins for the histogram.
        output_dir (str, optional): The directory to save the plot in.
    """
    plt.figure(figsize=(12, 7))

    for case in case_list:
        tau = case['tau']
        beta = case['beta']
        le_values = case['data'].flatten()
        
        # Plot the histogram for this specific case
        # Using alpha makes overlapping distributions easier to see
        plt.hist(le_values, bins=bins, alpha=0.6, density=True,
                 label=f'τ={tau:.2f}, β={beta/np.pi:.2f}π')

    # Use a logarithmic scale for the y-axis to see the full distribution
    plt.yscale('log')

    # Add labels and a title for the combined plot
    plt.title(f'Distribution of Lyapunov Exponents for $A_2 = {a2_value:.2f}$', fontsize=16)
    plt.xlabel('Lyapunov Exponent ($\lambda$)', fontsize=14)
    plt.ylabel('Probability Density', fontsize=14)
    plt.legend(title='Parameters')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"distribution_A2_{a2_value:.4f}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Grouped distribution plot saved to: {save_path}")

# --- MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    # Directory where your MLE csv files are stored
    data_dir = '../MLE' 
    plot_dir = '../Plot'

    # 1. Find all MLE files and group them by A2 value
    data_groups = defaultdict(list)
    filename_pattern = re.compile(r'A2_(\d+\.?\d*)_tau_(\d+\.?\d*)_beta_(\d+\.?\d*)\.csv$')

    try:
        all_files = sorted(os.listdir(data_dir), key=natural_sort_key)
    except FileNotFoundError:
        print(f"Error: Directory '{data_dir}' not found. Exiting.")
        exit()

    for filename in all_files:
        match = filename_pattern.match(filename)
        if match:
            try:
                # Parse parameters from filename
                a2 = float(match.group(1))
                tau = float(match.group(2))
                beta = float(match.group(3))
                
                # Load the data
                full_path = os.path.join(data_dir, filename)
                data = np.loadtxt(full_path)
                
                # Add the loaded data and its parameters to the correct group
                data_groups[a2].append({'tau': tau, 'beta': beta, 'data': data})
                print(f"Loaded and grouped '{filename}'")
            except Exception as e:
                print(f"Could not process file {filename}: {e}")

    # 2. Loop through each A2 group and create a plot
    if not data_groups:
        print("No data files were found or grouped. Please check your filenames and directory.")
    else:
        print("\n--- Starting to generate plots for each A2 group ---")
        for a2_val, cases in data_groups.items():
            plot_grouped_distributions(a2_val, cases, output_dir=plot_dir)