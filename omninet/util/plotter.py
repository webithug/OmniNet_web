import numpy as np
import matplotlib.pyplot as plt
import os

def plot_and_save_histograms(data: dict[str, np.ndarray], base_dir: str = "histograms") -> None:
    """
    Plots and saves histograms for each NumPy array in the input dictionary.

    Parameters:
    - data: A dictionary where the keys are strings and the values are NumPy arrays.
    - base_dir: The base directory where the histogram images will be saved. Default is "histograms".
    """
    # Ensure the base directory exists
    os.makedirs(base_dir, exist_ok=True)

    for key, array in data.items():
        # Create a specific directory for each key
        dir_path = os.path.join(base_dir, 'gen_distribution')
        os.makedirs(dir_path, exist_ok=True)

        # Create the histogram
        plt.figure(figsize=(10, 5))
        plt.hist(array, bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.title(f"Histogram for {key}")
        plt.xlabel(f"{key}")
        plt.ylabel('Frequency')

        # Save the figure
        save_path = os.path.join(dir_path, f"{key}_histogram.png")
        plt.savefig(save_path)
        plt.close()

        print(f"Histogram for '{key}' saved to {save_path}")


def plot_and_save_comparison_histograms(data: dict[str, np.ndarray],
                                        reference: dict[str, np.ndarray],
                                        base_dir: str = "histogram_comparisons") -> None:
    """
    Plots and saves comparison histograms for each data array against a reference array.

    Parameters:
    - data: A dictionary where the keys are strings and the values are NumPy arrays.
    - reference: A dictionary where the keys are strings matching those in `data` and the values are reference NumPy arrays.
    - base_dir: The base directory where the comparison histogram images will be saved. Default is "histogram_comparisons".
    """
    # Ensure the base directory exists
    os.makedirs(base_dir, exist_ok=True)

    for key, data_array in data.items():
        ref_array = reference.get(key)
        if ref_array is None:
            print(f"No reference data for key '{key}', skipping.")
            continue

        # Create a specific directory for each key
        dir_path = os.path.join(base_dir, 'comparison')
        os.makedirs(dir_path, exist_ok=True)

        # Determine the common range for x-axis
        combined_data = np.concatenate([data_array, ref_array])
        min_val, max_val = combined_data.min(), combined_data.max()

        # Plot the histograms with a shared x-axis range
        plt.figure(figsize=(10, 5))
        plt.hist(data_array, bins=30, alpha=0.5, color='blue', edgecolor='black',
                 density=True, label='Data', range=(min_val, max_val))
        plt.hist(ref_array, bins=30, alpha=0.5, color='red', edgecolor='black',
                 density=True, label='Reference', range=(min_val, max_val))

        # Add title and labels
        plt.title(f"Comparison Histogram for {key}")
        plt.xlabel(f"{key}")
        plt.ylabel('Normalized Frequency')
        plt.legend()

        # Save the figure
        save_path = os.path.join(dir_path, f"{key}_comparison_histogram.png")
        plt.savefig(save_path)
        plt.close()

        print(f"Comparison histogram for '{key}' saved to {save_path}")
