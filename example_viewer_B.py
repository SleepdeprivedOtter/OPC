
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt

path = "./"
image_to_view = "\Results\output_model15.fits"
image_path = path + image_to_view

image = np.array(fits.getdata(image_path))
hdul = fits.open(image_path)
hdul.info()

# Load the images
residual = hdul[3].data
original = hdul[1].data
model = hdul[2].data

# Plot the images
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(original, cmap='gray', origin='lower')
axs[0].set_title("Original Image")
axs[1].imshow(model, cmap='gray', origin='lower')
axs[1].set_title("Model Image")
axs[2].imshow(residual, cmap='gray', origin='lower')
axs[2].set_title("Residual Image")
plt.show()


