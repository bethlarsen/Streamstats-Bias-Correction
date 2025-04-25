import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


# Define possible function types
def power_law(x, a, b):
    return a * np.power(x, b)


def exponential(x, a, b):
    return a * np.exp(b * x)


def logarithmic(x, a, b):
    return a + b * np.log(x)


def polynomial(x, a, b, c):
    return a * x ** 2 + b * x + c

def linear(x, a, b):
    return a * x + b


# Load CSV file
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    return df


# Fit curve and evaluate best model
def fit_best_curve(percentiles, flows):
    functions = {
        "Power Law": power_law,
        "Exponential": exponential,
        "Logarithmic": logarithmic,
        "Polynomial": polynomial,
        "Linear": linear
    }

    best_fit = None
    best_r2 = -np.inf
    best_params = None
    best_func = None

    for name, func in functions.items():
        try:
            params, _ = curve_fit(func, percentiles, flows, maxfev=5000)
            predicted = func(percentiles, *params)
            r2 = r2_score(flows, predicted)

            if r2 > best_r2:
                best_r2 = r2
                best_fit = name
                best_params = params
                best_func = func
        except Exception as e:
            print(f"Error fitting {name} model: {e}")
        continue


    return best_fit, best_func, best_params, best_r2


# Generate equation string
def get_equation(best_fit, best_params):
    if best_fit == "Power Law":
        return f"Flow = {best_params[0]:.4f} * Percentile^{best_params[1]:.4f}"
    elif best_fit == "Exponential":
        return f"Flow = {best_params[0]:.4f} * exp({best_params[1]:.4f} * Percentile)"
    elif best_fit == "Logarithmic":
        return f"Flow = {best_params[0]:.4f} + {best_params[1]:.4f} * log(Percentile)"
    elif best_fit == "Polynomial":
        return f"Flow = {best_params[0]:.4f} * Percentile^2 + {best_params[1]:.4f} * Percentile + {best_params[2]:.4f}"
    elif best_fit == "Linear":
        return f"Flow = {best_params[0]:.4f} + {best_params[1]:.4f}"

    else:
        return "No equation available."


# Generate predicted values and save to CSV
def save_predicted_values(monthly_models, output_file="/Users/beth/Documents/Bias Correction/Test Files/predicted_fdc_monthly.csv"):
    exceedance_probs = np.concatenate((np.arange(0.1, 1.1, 0.1), np.arange(2, 101)))

    df_list = []
    for month, (best_func, best_params) in monthly_models.items():
        predicted_flows = best_func(exceedance_probs, *best_params)
        df_temp = pd.DataFrame({"Exceedance Probability": exceedance_probs, f"Flow_{month}": predicted_flows})
        df_list.append(df_temp.set_index("Exceedance Probability"))

    df_final = pd.concat(df_list, axis=1).reset_index()
    df_final.to_csv(output_file, index=False)
    print(f"Predicted values saved to {output_file}")


# Plot the results
def plot_fdc(percentiles, flows, best_func, best_params, best_fit, month):
    plt.scatter(percentiles, flows, label="Observed Data", color='blue')
    plt.plot(percentiles, best_func(percentiles, *best_params), label=f"Best Fit: {best_fit}", color='red')
    plt.xlabel("Percent Exceedance")
    plt.ylabel("Flow")
    plt.title(f"Flow Duration Curve - {month}")
    plt.legend()
    plt.grid()
    plt.show()


# Main function
def main(csv_file):
    df = load_data(csv_file)
    months = [col for col in df.columns if col not in ["Percentile"] and "Unnamed" not in col]
    monthly_models = {}

    for month in months:
        percentiles = df["Percentile"].values
        flows = pd.to_numeric(df[month], errors="coerce")  # Ensure numeric

        # Filter out any rows where either is NaN
        mask = ~np.isnan(percentiles) & ~np.isnan(flows)
        percentiles = percentiles[mask]
        flows = flows[mask]

        if len(percentiles) == 0 or len(flows) == 0:
            print(f"No valid data for {month}.")
            continue
        best_fit, best_func, best_params, best_r2 = fit_best_curve(percentiles, flows)

        if best_fit:
            equation = get_equation(best_fit, best_params)
            print(f"Month: {month}")
            print(f"Best Fit Model: {best_fit}")
            print(f"Equation: {equation}")
            print(f"Parameters: {best_params}")
            print(f"R² Score: {best_r2:.4f}")
            plot_fdc(percentiles, flows, best_func, best_params, best_fit, month)
            monthly_models[month] = (best_func, best_params)
        else:
            print(f"No suitable model found for {month}.")

    save_predicted_values(monthly_models)


# Run script
if __name__ == "__main__":
    csv_filename = "/Users/beth/Documents/Bias Correction/Penobscot River Data/Penobscot_monthly_fdc_vals.csv"  # Change this to your CSV file name
    main(csv_filename)
