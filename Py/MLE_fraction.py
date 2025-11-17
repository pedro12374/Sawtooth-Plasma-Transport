import os
import re
import numpy as np
import pandas as pd

# --- UTILITY FUNCTIONS ---

def natural_sort_key(filename):
    """A key function for natural sorting of strings containing numbers."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

def calculate_chaotic_fraction(data, threshold=1e-4):
    """
    Calculates the chaotic fraction from a loaded MLE data array.
    """
    if data.size == 0:
        return 0.0
    data_m = data>0
    chaotic_pixels = np.sum(data_m)
    return chaotic_pixels / data.size

def load_csv_files_with_three_values(directory):
    """
    Loads CSV files from a directory, parsing A2, tau, and beta from the filename.
    """
    f_data = []
    extracted_triplets = []
    # Regex to match your C++ output filenames
    filename_pattern = re.compile(r'A2_(\d+\.?\d*)_tau_(\d+\.?\d*)_beta_(\d+\.?\d*)\.csv$')

    try:
        all_items_in_directory = os.listdir(directory)
    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.")
        return [], []

    csv_basenames = sorted([f for f in all_items_in_directory if filename_pattern.match(f)], key=natural_sort_key)

    for basename in csv_basenames:
        match = filename_pattern.match(basename)
        if match:
            val1_str, val2_str, val3_str = match.groups()
            try:
                val1 = float(val1_str) # A2
                val2 = float(val2_str) # tau
                val3 = float(val3_str) # beta
                full_file_path = os.path.join(directory, basename)
                data_array = np.loadtxt(full_file_path)
                print(f"Loaded data from '{basename}'...")
                
                f_data.append(data_array)
                extracted_triplets.append([val1, val2, val3])
            except Exception as e:
                print(f"Warning: Could not load or parse file '{basename}': {e}. Skipping.")
            
    return f_data, extracted_triplets

# --- MAIN EXECUTION BLOCK ---

# 1. SETUP
csv_directory = "../MLE"  # Directory where your MLE csv files are stored

# 2. LOAD DATA
vetor, name_parameter_triplets = load_csv_files_with_three_values(csv_directory)
analysis_results = [] 

if not vetor:
    print("No matching data files were found. Please check directory and filenames.")
else:
    print(f"\nSuccessfully loaded {len(vetor)} files. Starting quantitative analysis...")

    # 3. PROCESS EACH FILE
    for i, data in enumerate(vetor):
        # Get the corresponding [A2, tau, beta] triplet
        A2, tau, beta = name_parameter_triplets[i]
        
        # Calculate the chaotic fraction for the loaded data
        fraction = calculate_chaotic_fraction(data)
        print(f"  -> Parameters: A2={A2}, tau={tau}, beta={beta} -> Chaotic Fraction: {fraction:.4f}")
        
        # Store the result for the final summary
        analysis_results.append({'A2': A2, 'tau': tau, 'beta': beta, 'chaotic_fraction': fraction})

# 4. SAVE FINAL SUMMARY
if analysis_results:
    df = pd.DataFrame(analysis_results)
    print("\n--- Quantitative Analysis Summary ---")
    print(df)
    
    # Save the summary data to a CSV file for future use
    df.to_csv("chaotic_fraction_summary.csv", index=False)
    print("\nSummary of chaotic fractions saved to chaotic_fraction_summary.csv")