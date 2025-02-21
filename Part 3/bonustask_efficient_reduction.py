import glob; import numpy as np; from astropy.io import fits; import os
# Defining the path to the files
path_files = r"C:\Users\96fan\Desktop\OPC\Part 3\Data\H"
# Reading the darks for flats
darks_for_flats = np.array([fits.getdata(path) for path in glob.glob(path_files + "\\*DARK_DIT3.0_NDIT1*.fits")])
# Write out the master dark for flats
fits.writeto(f"{path_files}/masterdark_for_flats.fits", data=np.median(darks_for_flats, axis=0), header=None, overwrite=True)
# Reading the darks for science
darks_for_science = np.array([fits.getdata(path) for path in glob.glob(path_files + "\\*DARK_DIT2.0_NDIT25*.fits")])
# Write out the master dark for science
fits.writeto(f"{path_files}/masterdark_for_science.fits", data=np.median(darks_for_science, axis=0), header=None, overwrite=True)
# Reading the flats
flats = np.array([fits.getdata(path) - np.median(darks_for_flats, axis=0) for path in glob.glob(path_files + "\\*FLAT_DIT3.0_NDIT1*.fits")])
# Normalize the flats and write out the master flat
master_flat = np.median(np.array([f/np.nanmedian(f) for f in flats]), axis=0)
fits.writeto(f"{path_files}/masterflat.fits", data=master_flat, header=None, overwrite=True)
# Reading the science images, subtracting the darks, dividing by the master flat
science = np.array([fits.getdata(path) - np.median(darks_for_science, axis=0) for path in glob.glob(path_files + "\\*SCIENCE_DIT2.0_NDIT25*.fits")])/ master_flat
# Subtracting the background of individual images
science = np.array([s-np.nanmedian(s) for s in science])
# Writing out the background image and subtracting the background of the images
fits.writeto(f"{path_files}/background.fits", data=np.nanmedian(science, axis=0), header=None, overwrite=True)
science = science - np.nanmedian(science, axis=0)
# Remove the stripping pattern
row_medians = np.expand_dims(np.nanmedian(science,axis=1),axis=1)
science = science - row_medians + np.nanmedian(science)
# Write out the final science images
for i, sci in enumerate(science):
    header = fits.getheader(glob.glob(path_files + "\\*SCIENCE_DIT2.0_NDIT25*.fits")[i])
    fits.writeto(f"{path_files}/science_final_destripped_{i+1}.fits", data=sci, header=header, overwrite=True)



