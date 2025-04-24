# Streamstats-Bias-Correction
Scripts to help in the analysis of bias correction methods from linear regression equations from Streamstats. 

## Correct FDC Shifting
This script takes in simulated, observed, and FDC values csv and shifts the simulated values based on the best factor from the FDC values. All the FDCs are plotted for the sake of comparison.
This is for a singular, total FDC, not monthly. 

## Correct FDC Scaling
This script does the same as correct_FDC_shift, except it scales the values by a factor.
It's often helpful to add or remove log axes in order to aid in comparison of the curves.

