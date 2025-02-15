
from astropy import units as u
import numpy as np
from astropy.nddata import CCDData
import ccdproc
import glob
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

import warnings
warnings.filterwarnings("ignore")

path = "./"

bias_data = glob.glob(path + "*bias*")
bias = np.array([fits.getdata(data) for data in bias_data])

flat_data = glob.glob(path + "*flat*")
flat = np.array([fits.getdata(data) for data in flat_data])


#masterbias
combiner = ccdproc.Combiner([CCDData(bias[0], unit=u.adu),CCDData(bias[1], unit=u.adu), CCDData(bias[2], unit=u.adu),CCDData(bias[3], unit=u.adu), CCDData(bias[4], unit=u.adu),CCDData(bias[5], unit=u.adu), CCDData(bias[6], unit=u.adu)])
master_bias = combiner.average_combine()
#fits.writeto("./master_bias.fits", data=np.array(master_bias) , header=None)
print(f"The mean bias-level is: {np.mean(np.array(master_bias)):.2f} ADU")
print(f"The median bias-level is: {np.median(np.array(master_bias)):.2f} ADU")
print(f"The standard deviation of the bias-level is: {np.std(np.array(master_bias)):.2f} ADU")


# List of flat image pairs 
flat_pairs = [("flat0002.fits", "flat0003.fits"),
              ("flat0004.fits", "flat0005.fits"),
              ("flat0006.fits", "flat0007.fits"),
              ("flat0008.fits", "flat0009.fits"),
              ("flat0010.fits", "flat0011.fits"),
              ("flat0012.fits", "flat0013.fits"),
              ("flat0014.fits", "flat0015.fits")]

# Lists to store mean signal and variance values
mean_signals = []
variances = []

# Loop through all flat field pairs
for flat1_filename, flat2_filename in flat_pairs:
    # Read the flat images
    flat1 = CCDData.read(flat1_filename, unit="adu")
    flat2 = CCDData.read(flat2_filename, unit="adu")

    # Compute mean signal
    mean_signal = (np.mean(flat1.data) + np.mean(flat2.data)) / 2
    mean_signals.append(mean_signal)

    # Compute variance using the difference method
    variance = np.sum((flat1.data - flat2.data) ** 2) / (2 * flat1.size)
    variances.append(variance)

# Convert to numpy arrays
mean_signals = np.array(mean_signals)
variances = np.array(variances)

gains= []
for i in range(len(mean_signals)):
    # Compute the gain (G = mean_signal / variance)
    gain = mean_signals[i] / variances[i]
    gains.append(gain)

    print(f"Mean Signal: {mean_signals[i]:.2f} ADU")
    print(f"Mean Variance: {variances[i]:.2f} ADU²")
    print(f"Camera Gain from single image pair: {gain:.2f} e-/ADU")


# Lists to store mean signal and variance values
mean_signals = []
variances = []

# Loop through all flat field pairs
for flat1_filename, flat2_filename in flat_pairs:
    # Read the flat images
    flat1 = CCDData.read(flat1_filename, unit="adu")
    flat2 = CCDData.read(flat2_filename, unit="adu")

    # Compute mean signal
    mean_signal = (np.mean(flat1.data) + np.mean(flat2.data)) / 2
    mean_signals.append(mean_signal)

    # Compute variance using the difference method
    variance = np.sum((flat1.data - flat2.data) ** 2) / (2 * flat1.size)
    variances.append(variance)

# Convert to numpy arrays
mean_signals = np.array(mean_signals)
variances = np.array(variances)

# Define a linear function for fitting
def linear_model(x, a, b):
    return a * x + b  # y = a * x, where a is the gain (e-/ADU)

# Fit the data in the linear region (exclude potential saturation)
linear_region = mean_signals < 0.5 * np.max(mean_signals)  # Select points below 50% max signal
popt, pcov = curve_fit(linear_model, mean_signals[linear_region], variances[linear_region])

# Extract the gain from the slope
fitted_gain = popt[0]  # Gain in e-/ADU
gain_uncertainty = np.sqrt(np.diag(pcov))[0]  # Uncertainty in gain

# Generate ideal variance values for plotting
ideal_variances = linear_model(mean_signals, fitted_gain, popt[1])

# 📊 Plot the Photon Transfer Curve with Fit
plt.figure(figsize=(8,6))
plt.scatter(mean_signals, variances, label="Measured Data", color='blue')
plt.plot(mean_signals, ideal_variances, linestyle="dashed", color='red', label=f"Linear Fit (Gain = {fitted_gain:.2f} e-/ADU)")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Mean Signal (ADU)")
plt.ylabel("Variance (ADU²)")
plt.title("Photon Transfer Curve (PTC) with Linear Fit")
plt.legend()
plt.grid(True, which="both", linestyle="--")
plt.show()

# Print the extracted gain
print(f"Camera Gain from PTC: {fitted_gain:.2f} ± {gain_uncertainty:.2f} e-/ADU")


# Read the master bias frame
master_bias = CCDData.read("master_bias.fits", unit="adu")

# Compute the standard deviation of the master bias
sigma_bias = np.std(master_bias.data)
print(f"Standard Deviation of Master Bias: {sigma_bias:.2f} ADU")

# Compute readout noise in e-
readout_noise = fitted_gain * sigma_bias
print(f"Readout Noise: {readout_noise:.2f} e⁻")

from scipy.optimize import curve_fit

# Define a linear function for fitting
def linear_model(x, a):
    return a * x  # Linear model y = a * x

# Perform linear fit using data from the lower (linear) region
linear_region = mean_signals < 0.5 * np.max(mean_signals)  # Select points in the linear regime
popt, _ = curve_fit(linear_model, mean_signals[linear_region], variances[linear_region])

# Compute the ideal variance based on the linear fit
ideal_variances = linear_model(mean_signals, *popt)

# Compute INL percentage
INL = ((variances - ideal_variances) / ideal_variances) * 100

print(mean_signals)

# Plot the INL
plt.figure(figsize=(8,6))
plt.plot(mean_signals, INL, marker="o", linestyle="-", label="INL")
plt.axhline(0, color="gray", linestyle="--")
plt.xlabel("Mean Signal (ADU)")
plt.ylabel("INL (%)")
plt.title("Integral Non-Linearity (INL)")
plt.legend()
plt.grid(True)
plt.show()

# Print the maximum absolute INL
max_inl = np.max(np.abs(INL))
print(f"Maximum INL: {max_inl:.2f}%")


