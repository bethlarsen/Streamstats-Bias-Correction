import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

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

    if prob_column not in df.columns or flow_column not in df.columns:
        raise ValueError(f"Columns '{prob_column}' and/or '{flow_column}' not found in {csv_file}.")

    return df[prob_column], df[flow_column]

def find_best_shift(simulated_flows, precomputed_flows, simulated_probs, precomputed_probs):
    """Finds the best additive shift to align simulated FDC with precomputed FDC."""

    # Convert to NumPy arrays
    simulated_flows = np.array(simulated_flows).flatten()
    precomputed_flows = np.array(precomputed_flows).flatten()
    simulated_probs = np.array(simulated_probs).flatten()
    precomputed_probs = np.array(precomputed_probs).flatten()

    # Interpolate precomputed flows to match exceedance probabilities of simulated flows
    precomputed_flows = np.interp(simulated_probs, precomputed_probs, precomputed_flows)

    # Define the middle range for optimization (30% - 80% exceedance probability)
    lower_bound = 30
    upper_bound = 80
    valid_indices = (simulated_probs >= lower_bound) & (simulated_probs <= upper_bound)

    # Select values only within the chosen exceedance probability range
    sim_flows_subset = simulated_flows[valid_indices]
    precomp_flows_subset = precomputed_flows[valid_indices]

    # Remove NaNs
    valid_indices = ~np.isnan(sim_flows_subset) & ~np.isnan(precomp_flows_subset)
    sim_flows_subset = sim_flows_subset[valid_indices]
    precomp_flows_subset = precomp_flows_subset[valid_indices]

    def error_function(shift):
        """Minimizes the difference between adjusted simulated flows and precomputed FDC."""
        adjusted_simulated = sim_flows_subset + shift
        return np.sum((adjusted_simulated - precomp_flows_subset) ** 2)

    # Find best shift value
    result = minimize(error_function, x0=[0.0], bounds=[(-100, 100)])  # Adjust bounds as needed
    best_shift = result.x[0] if result.success else 0.0  # Default to no shift if optimization fails

    return best_shift

# Load data
simulated = pd.read_csv("/Users/beth/Documents/Bias Correction/Eagle Creek Data/simulated_data_EagleCreek.csv")["Flow"]
observed = pd.read_csv("/Users/beth/Documents/Bias Correction/Eagle Creek Data/Eagle_Creek Gage data.csv")["Flow"]
precomputed_fdc_file = "/Users/beth/Documents/Bias Correction/Eagle Creek Data/Eagle Creek FDC.csv"

simulated_sorted = np.sort(np.array(simulated).flatten())[::-1]  # Sort descending
observed_sorted = np.sort(np.array(observed).flatten())[::-1]

# Compute FDCs
simulated_probs, simulated_fdc = compute_fdc(simulated)
observed_probs, observed_fdc = compute_fdc(observed)
precomputed_probs, precomputed_fdc = load_precomputed_fdc(precomputed_fdc_file)

# Find optimal shift and apply correction
shift = find_best_shift(simulated_fdc, precomputed_fdc, simulated_probs, precomputed_probs)
adjusted_simulated = simulated_fdc + shift

print(f"Optimal Shift: {shift}")

# Plot comparison
plt.figure(figsize=(8, 7))
#plt.plot(simulated_probs, simulated_fdc, label="Original Simulated", linestyle="--")
plt.plot(simulated_probs, adjusted_simulated, label="Shifted Simulated", linestyle="-")
#plt.plot(precomputed_probs, precomputed_fdc, label="Precomputed FDC", linestyle=":")
plt.plot(observed_probs, observed_fdc, label="Observed", linestyle=":")
#plt.xscale("log")
#plt.yscale("log")
plt.xlabel("Exceedance Probability (%)")
plt.ylabel("Streamflow (m³/s)")
plt.title("Flow Duration Curves Comparison- Vertical Shift")
plt.legend()
plt.grid()
plt.show()
