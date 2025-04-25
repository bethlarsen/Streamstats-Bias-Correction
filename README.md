# Streamstats-Bias-Correction
Scripts to help in the analysis of bias correction methods based on linear regression equations from Streamstats. 

## Correct FDC Shifting
This script takes in simulated, observed, and FDC values csv and shifts the simulated values based on the best factor from the FDC values. All the FDCs are plotted for the sake of comparison.
This is for a singular, total FDC, not monthly. 

## Correct FDC Scaling
This script does the same as correct_FDC_shift, except it scales the values by a factor.
It's often helpful to add or remove log axes in order to aid in comparison of the curves.

## Fit FDC Equation
This script takes in a csv of exceedance probabilities and flow values, fits a regression equation to these values, and outputs a csv with all the exceedance probabilities from 0.1-100 with the
corresponding values based on the fitted equation. 

## Fit FDC Equation Monthly
This script does the same as fit_FDC_equation, except it takes in a csv with a column for exceedance probabilities and twelve columns, named by month, of flow values. 

