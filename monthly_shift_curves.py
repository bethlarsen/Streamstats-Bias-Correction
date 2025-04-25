import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def load_predicted_fdc_monthly(csv_path):
    df = pd.read_csv(csv_path)
    exceedance_probs = df['Exceedance Probability']
    monthly_flows = {col.replace('Flow_', ''): df[col].values for col in df.columns if 'Flow_' in col}
    return exceedance_probs, monthly_flows

def compute_fdc(flows):
    sorted_flows = np.sort(flows.dropna())[::-1]
    ranks = np.arange(1, len(sorted_flows) + 1)
    exceedance_probs = 100 * (ranks / (len(sorted_flows) + 1))
    return exceedance_probs, sorted_flows

def find_best_shift(simulated_flows, precomputed_flows, simulated_probs, precomputed_probs):
    simulated_flows = np.array(simulated_flows).flatten()
    precomputed_flows = np.interp(simulated_probs, precomputed_probs, precomputed_flows)

    valid = ~np.isnan(simulated_flows) & ~np.isnan(precomputed_flows)
    simulated_flows = simulated_flows[valid]
    precomputed_flows = precomputed_flows[valid]

    def error_function(shift):
        return np.sum((simulated_flows + shift - precomputed_flows) ** 2)

    result = minimize(error_function, x0=[0.0], bounds=[(-10000, 10000)])
    return result.x[0] if result.success else 0.0

def plot_monthly_fdcs(month_name, sim_probs, sim_flows, shifted_flows, precomp_probs, precomp_flows):
    plt.figure(figsize=(7, 5))
    plt.plot(sim_probs, sim_flows, label="Original Simulated", linestyle='--')
    plt.plot(sim_probs, shifted_flows, label="Shifted Simulated", linestyle='-')
    precomp_probs = precomp_probs[::-1]
    plt.plot(precomp_probs, precomp_flows, label="Precomputed", linestyle=':')
    #plt.xscale("log")
    #plt.yscale("log")
    plt.xlabel("Exceedance Probability (%)")
    plt.ylabel("Flow (m³/s)")
    plt.title(f"FDC Comparison - {month_name}")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.yscale('linear')
    plt.tight_layout()
    plt.show()

def apply_monthly_shifts(simulated_df, predicted_fdc_path):
    simulated_df['Month'] = simulated_df.index.month
    shifted_monthly = []

    precomp_probs, monthly_fdcs = load_predicted_fdc_monthly(predicted_fdc_path)

    for month_name, predicted_flows in monthly_fdcs.items():
        try:
            month_number = pd.to_datetime(month_name, format="%B").month
        except:
            print(f"Skipping invalid month column: {month_name}")
            continue

        monthly_sim = simulated_df[simulated_df['Month'] == month_number]['Flow']
        print(f"{month_name} flow sample:\n{monthly_sim.describe()}\n")
        sim_probs, sim_flows = compute_fdc(monthly_sim)
        shift = find_best_shift(sim_flows, predicted_flows, sim_probs, precomp_probs)

        print(f"{month_name}: Applying shift of {shift:.2f} m³/s")

        shifted = monthly_sim + shift
        shifted_monthly.append(shifted)

        shifted_fdc = sim_flows + shift
        plot_monthly_fdcs(month_name, sim_probs, sim_flows, shifted_fdc, precomp_probs, predicted_flows)

    return pd.concat(shifted_monthly).sort_index()




simulated = pd.read_csv("/Users/beth/Documents/Bias Correction/Penobscot River Data/Simulated_values_Penobscot.csv", index_col=0, parse_dates=True)

shifted = apply_monthly_shifts(simulated, "/Users/beth/Documents/Bias Correction/Test Files/predicted_fdc_monthly.csv")

observed = pd.read_csv("/Users/beth/Documents/Bias Correction/Penobscot River Data/Penobscot River Gage1938_1958.csv")["Flow"]
