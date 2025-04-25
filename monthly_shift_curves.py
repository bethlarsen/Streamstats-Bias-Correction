import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- Load precomputed monthly FDCs from CSV ---
def load_predicted_fdc_monthly(csv_path):
    df = pd.read_csv(csv_path)
    exceedance_probs = df['Exceedance Probability']

    monthly_flows = {}
    for col in df.columns:
        if col.startswith('Flow_'):
            # Extract month name from column headers like "Flow_January"
            month = col.split('_')[1]
            monthly_flows[month] = df[col].values

    return exceedance_probs, monthly_flows

# --- Compute Flow Duration Curve (FDC) from time series of flow values ---
def compute_fdc(flows):
    sorted_flows = np.sort(flows.dropna())[::-1]  # Sort flows in descending order
    ranks = np.arange(1, len(sorted_flows) + 1)   # Rank flows
    exceedance_probs = 100 * (ranks / (len(sorted_flows) + 1))  # Exceedance probabilities (%)
    return exceedance_probs, sorted_flows

# --- Find the optimal shift to align simulated flows to predicted FDCs ---
def find_best_shift(simulated_flows, precomputed_flows, simulated_probs, precomputed_probs):
    simulated_flows = np.array(simulated_flows).flatten()
    precomputed_flows = np.interp(simulated_probs, precomputed_probs, precomputed_flows)  # Interpolate precomputed flows to match simulated exceedance probs

    # Remove any NaNs
    valid = ~np.isnan(simulated_flows) & ~np.isnan(precomputed_flows)
    simulated_flows = simulated_flows[valid]
    precomputed_flows = precomputed_flows[valid]

    # Define error function: sum of squared differences between shifted simulated and precomputed flows
    def error_function(shift):
        return np.sum((simulated_flows + shift - precomputed_flows) ** 2)

    # Minimize the error function to find the best shift
    result = minimize(error_function, x0=[0.0], bounds=[(-10000, 10000)])
    return result.x[0] if result.success else 0.0

# --- Plot original, corrected, and precomputed FDCs for a given month ---
def plot_monthly_fdcs(month_name, sim_probs, sim_flows, corrected_flows,
                      precomp_probs, precomp_flows, obs_probs=None, obs_flows=None):
    plt.figure(figsize=(7, 5))
    plt.plot(sim_probs, sim_flows, label="Original Simulated", linestyle='--')
    plt.plot(sim_probs, corrected_flows, label="Corrected Simulated", linestyle='-')
    plt.plot(precomp_probs, precomp_flows[::-1], label="Precomputed", linestyle=':')

    # Optional: plot observed (gage) FDC if provided
    if obs_probs is not None and obs_flows is not None:
        plt.plot(obs_probs, obs_flows, label="Observed Gage", linestyle='-.', color='black')

    plt.xlabel("Exceedance Probability (%)")
    plt.ylabel("Flow (m³/s)")
    plt.title(f"FDC Comparison - {month_name}")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.yscale('log')  # Optional: logarithmic y-axis if flows vary widely
    plt.tight_layout()
    plt.show()

# --- Main function: apply monthly shifting to simulated flows ---
def apply_monthly_shifting(simulated_df, predicted_fdc_path, observed_df=None):
    simulated_df['Month'] = simulated_df.index.month  # Add Month column for filtering
    if observed_df is not None:
        observed_df['Month'] = observed_df.index.month

    shifted_monthly = []  # List to collect shifted monthly flow series
    precomp_probs, monthly_fdcs = load_predicted_fdc_monthly(predicted_fdc_path)

    for month_name, predicted_flows in monthly_fdcs.items():
        try:
            month_number = pd.to_datetime(month_name, format="%B").month  # Convert month name to month number
        except Exception as e:
            print(f"Could not parse month: {month_name} — {e}")
            continue

        # Select simulated flows for the current month
        monthly_sim = simulated_df[simulated_df['Month'] == month_number]['Flow']
        sim_probs, sim_flows = compute_fdc(monthly_sim)

        # (Optional) Select observed flows for the month if available
        obs_probs, obs_flows = None, None
        if observed_df is not None:
            monthly_obs = observed_df[observed_df['Month'] == month_number]['Flow']
            obs_probs, obs_flows = compute_fdc(monthly_obs)

        # Find best shift and apply it
        shift = find_best_shift(sim_flows, predicted_flows, sim_probs, precomp_probs)
        print(f"{month_name}: Applying shift of {shift:.2f} m³/s")

        shifted = monthly_sim + shift
        shifted_monthly.append(shifted)

        # For plotting: apply shift to original FDC
        corrected_fdc = sim_flows + shift
        plot_monthly_fdcs(month_name, sim_probs, sim_flows, corrected_fdc,
                          precomp_probs, predicted_flows,
                          obs_probs, obs_flows)

    return pd.concat(shifted_monthly).sort_index()  # Combine all months and sort by datetime

# --- Load observed and simulated data ---
observed = pd.read_csv("/Users/beth/Documents/Bias Correction/Penobscot River Data/Penobscot River Gage1938_1958.csv")
observed['Date'] = pd.to_datetime(observed['Date'])  # Convert Date column to datetime
observed.set_index('Date', inplace=True)  # Set Date as index

simulated = pd.read_csv("/Users/beth/Documents/Bias Correction/Penobscot River Data/Simulated_values_Penobscot.csv", index_col=0, parse_dates=True)

# --- Run monthly shifting correction ---
shifted = apply_monthly_shifting(simulated, "/Users/beth/Documents/Bias Correction/Test Files/predicted_fdc_monthly.csv", observed)
