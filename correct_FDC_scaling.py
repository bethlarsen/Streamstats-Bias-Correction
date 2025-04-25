import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def find_best_scaling_factor(simulated_flows, precomputed_flows, simulated_probs, precomputed_probs):
    """Finds the best scaling factor by comparing only the middle 30-80% exceedance range."""

    # Convert to NumPy arrays
    simulated_flows = np.array(simulated_flows).flatten()
    precomputed_flows = np.array(precomputed_flows).flatten()
    simulated_probs = np.array(simulated_probs).flatten()
    precomputed_probs = np.array(precomputed_probs).flatten()

    # Interpolate precomputed flows to match the exceedance probabilities of simulated flows
    precomputed_flows = np.interp(simulated_probs, precomputed_probs, precomputed_flows)

    # Define range (30% to 80% exceedance probability, or whatever you want to try)
    lower_bound, upper_bound = 30, 80
    valid_indices = (simulated_probs >= lower_bound) & (simulated_probs <= upper_bound)

    # Select only middle range of flows
    selected_simulated_flows = simulated_flows[valid_indices]
    selected_precomputed_flows = precomputed_flows[valid_indices]

    # Remove NaNs (in case any exist after filtering)
    valid_indices = ~np.isnan(selected_simulated_flows) & ~np.isnan(selected_precomputed_flows)
    selected_simulated_flows = selected_simulated_flows[valid_indices]
    selected_precomputed_flows = selected_precomputed_flows[valid_indices]

    # Handle zero/negative values for log function
    small_value = 1e-6
    selected_simulated_flows = np.maximum(selected_simulated_flows, small_value)
    selected_precomputed_flows = np.maximum(selected_precomputed_flows, small_value)

    def error_function(factor):
        """Error function to minimize: Difference between log-scaled FDCs (only middle range)."""
        scaled_simulated = selected_simulated_flows * factor
        return np.sum((np.log(scaled_simulated) - np.log(selected_precomputed_flows)) ** 2)

    # Find best scaling factor
    result = minimize(error_function, x0=[1.0], bounds=[(0.1, 10)])
    best_factor = result.x[0] if result.success else 1.0  # Default to 1 if optimization fails

    return best_factor  # Apply this factor to the full dataset outside the function


def compute_fdc(flow_data):
    """
    Computes the Flow Duration Curve (FDC) from a Pandas Series of streamflow data.
    """
    sorted_flows = np.sort(flow_data.dropna())[::-1]  # Sort in descending order
    ranks = np.arange(1, len(sorted_flows) + 1)
    exceedance_prob = 100 * (ranks / (len(sorted_flows) + 1))

    return exceedance_prob, sorted_flows


def load_precomputed_fdc(csv_file, prob_column='Exceedance Probability', flow_column='Flow'):
    """
    Reads a precomputed flow duration curve dataset and returns exceedance probabilities and flows.
    """
    df = pd.read_csv(csv_file)
    #df["Flow"] = df["Flow"][::-1]

    # Flip the Exceedance Probability column to match the sorted Flow values (comment the following two lines in or out)
    #df = df.sort_values(by="Flow", ascending=False).reset_index(drop=True)
    #df["Exceedance Probability"] = df["Exceedance Probability"][::-1].values

    if prob_column not in df.columns or flow_column not in df.columns:
        raise ValueError(f"Columns '{prob_column}' and/or '{flow_column}' not found in {csv_file}.")

    return df[prob_column], df[flow_column]



def plot_multiple_fdcs(time_series_files, precomputed_fdc_file, labels):
    """
    Plots flow duration curves for two time-series datasets and one precomputed FDC.

    Args:
        time_series_files (list): List of two CSV file paths (e.g., [observed.csv, simulated.csv])
        precomputed_fdc_file (str): File path of precomputed FDC dataset
        labels (list): Labels for the three datasets in the plot
    """
    if len(time_series_files) != 3 or len(labels) != 4:
        raise ValueError(
            "Provide exactly two time-series files and three labels (for both time-series and precomputed FDC).")

    # Compute FDCs for the two time-series datasets
    prob1, flow1 = compute_fdc(time_series_files[0])
    prob2, flow2 = compute_fdc(time_series_files[1])
    prob4, flow4 = compute_fdc(time_series_files[2])

    # Load the precomputed FDC dataset
    prob3, flow3 = load_precomputed_fdc(precomputed_fdc_file)

    # Plot FDCs
    plt.figure(figsize=(8, 6))
    plt.plot(prob1, flow1, marker='o', linestyle='-', markersize=1, label=labels[0])
    plt.plot(prob2, flow2, marker='s', linestyle='--', markersize=1, label=labels[1])
    #plt.plot(prob3, flow3, marker='d', linestyle=':', markersize=1, label=labels[2])
    #plt.plot(prob4, flow4, marker='o', linestyle='-', markersize=1, label=labels[3])

    # Formatting plot
    plt.xlabel("Exceedance Probability (%)")
    plt.ylabel("Streamflow (m³/s)")
    plt.title("Flow Duration Curves Comparison")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    #plt.xscale("log")
    #plt.yscale("log")
    #plt.gca().invert_xaxis()  # Reverse X-axis
    plt.legend()

    # Show plot
    plt.show()

observed = pd.read_csv(
    "/Users/beth/Documents/Bias Correction/Eagle Creek Data/Eagle_Creek Gage data.csv",
    index_col=0,
    parse_dates=True
)["Flow"]

simulated = pd.read_csv("/Users/beth/Documents/Bias Correction/Eagle Creek Data/simulated_data_EagleCreek.csv", index_col=0, parse_dates=True)["Flow"]

fdc = pd.read_csv("/Users/beth/Documents/Bias Correction/Eagle Creek Data/Eagle Creek FDC.csv", index_col=0)["Flow"]

fdc_prob = pd.read_csv("/Users/beth/Documents/Bias Correction/Eagle Creek Data/Eagle Creek FDC.csv")["Exceedance Probability"]

fdc2 = "/Users/beth/Documents/Bias Correction/Eagle Creek Data/Eagle Creek FDC.csv"

prob, simulated_fdc = compute_fdc(simulated)

factor = find_best_scaling_factor(simulated_fdc, fdc, prob, fdc_prob)
simulated_adj = simulated.apply(lambda x: x * factor)

print(factor)

# Example usage:
plot_multiple_fdcs([observed, simulated_adj, simulated], fdc2,
                   ["Observed", "Simulated Adjusted", "Precomputed FDC", "Simulated"])
