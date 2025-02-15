
from astropy import units as u
import numpy as np
from astropy.nddata import CCDData
import ccdproc
import glob
from astropy.io import fits


path = "./SN2012aw/"

bias_data = glob.glob(path + "*bias*")
bias = np.array([fits.getdata(data) for data in bias_data])

img_data = glob.glob(path + "*.SN.fits")
img = np.array([fits.getdata(data) for data in img_data])


#remove cosmics
img_clean = []
for i in img:
    clean = ccdproc.cosmicray_median(i, mbox=10,rbox=10, gbox=5)
    img_clean.append(clean)   


#masterbias
combiner = ccdproc.Combiner([CCDData(bias[0], unit=u.adu),CCDData(bias[1], unit=u.adu), CCDData(bias[2], unit=u.adu),CCDData(bias[3], unit=u.adu), CCDData(bias[4], unit=u.adu),CCDData(bias[5], unit=u.adu), CCDData(bias[6], unit=u.adu),CCDData(bias[7], unit=u.adu), CCDData(bias[8], unit=u.adu),CCDData(bias[9], unit=u.adu)])
master_bias = combiner.average_combine()

#masterdark
#exptime = 600
dark_data = glob.glob(path + "*dark*")
dark = np.array([fits.getdata(data) - master_bias for data in dark_data])

combiner = ccdproc.Combiner([CCDData(dark[0], unit=u.adu),CCDData(dark[1], unit=u.adu), CCDData(dark[2], unit=u.adu),CCDData(dark[3], unit=u.adu), CCDData(dark[4], unit=u.adu)])
master_dark = combiner.average_combine()

#masterflat
#exptime = 5
flat_data = glob.glob(path + "*flat*")
flat_mid = np.array([fits.getdata(data) - master_bias for data in flat_data])
flat = np.array([data*120 - master_dark for data in flat_mid])
flat_norm = np.array([flat/np.nanmedian(flat) for flat in flat])
combiner = ccdproc.Combiner([CCDData(flat[0], unit=u.adu),CCDData(flat[1], unit=u.adu), CCDData(flat[2], unit=u.adu),CCDData(flat[3], unit=u.adu), CCDData(flat[4], unit=u.adu),CCDData(flat[5], unit=u.adu), CCDData(flat[6], unit=u.adu),CCDData(flat[7], unit=u.adu), CCDData(flat[8], unit=u.adu),CCDData(flat[9], unit=u.adu)])
master_flat = combiner.average_combine()



#image reduction
img = []
exptimes = [900,900,900]
for t,i in enumerate(img_clean):
    ccd = CCDData(i[0], unit=u.adu)
    ccd = ccdproc.subtract_bias(ccd, master_bias)
    nccd = ccdproc.ccd_process(ccd, gain=0.68*u.adu/u.adu, dark_frame = master_dark, dark_exposure = 600*u.second, data_exposure = exptimes[t]*u.second, dark_scale = True, master_flat = master_flat)
    img.append(nccd)
combiner = ccdproc.Combiner([img[0], img[1], img[2]])
img = combiner.average_combine()
img = np.array(img)
print(np.median(img))
fits.writeto("./SN2012aw/img.fits", data=img , header=None, overwrite=True)


#image reduction of the landolt field
img_data = glob.glob(path + "*object*")
img = np.array([fits.getdata(data) for data in img_data])

#remove cosmics
img_clean = []
for i in img:
    clean = ccdproc.cosmicray_median(i, mbox=10,rbox=10, gbox=5)
    img_clean.append(clean)   

img = []
exptimes = [120,30]
for t,i in enumerate(img_clean):
    ccd = CCDData(i[0], unit=u.adu)
    ccd = ccdproc.subtract_bias(ccd, master_bias)
    nccd = ccdproc.ccd_process(ccd, gain=0.68*u.adu/u.adu, dark_frame = master_dark, dark_exposure = 600*u.second, data_exposure = exptimes[t]*u.second, dark_scale = True, master_flat = master_flat)
    img.append(nccd)
img1 = CCDData(img[0], unit=u.adu)
img2 = CCDData(img[1], unit=u.adu)
img2_data = img2.data * 4
img2 = CCDData(img2_data, unit=u.adu)
combiner = ccdproc.Combiner([img1, img2])
img = combiner.average_combine()
img_landolt = np.array(img)
fits.writeto("./SN2012aw/landolt.fits", data=img_landolt , header=None, overwrite=True)

#photometry
t = np.nanmedian(exptimes)
k = 0.127
A = np.nanmedian([1.358,1.255,1.243])

# Calculate m0 (photometric zero point)
def calc_m0(mr, Is, Ib, t, A):
    m0 = mr + 2.5*np.log10((Is - Ib)/t)+ k * A
    return m0

# Calculate mr (instrumental magnitude)
def calc_mr(m0, Is, Ib, t, A):
    mr = m0 - 2.5*np.log10((Is - Ib)/t)- k * A
    return mr

# Landolt stars instrumental magnitudes
mr_landolt = [9.539-0.009, 12.271-0.080, 13.385-0.575, 11.930-0.723, 13.068-0.683,
              13.398-1.082, 17.800-3.100, 13.749-0.366, 11.954-0.290]

# Landolt stars instrumental counts
Is_landolt = [21757, 4470, 2728, 5215, 3955, 4083, 770, 1932, 3914]

# Landolt background instrumental counts
Ib_landolt = np.median(img_landolt)

# Landolt exposure time
t_landolt = 120

# Landolt airmass
A_landolt = np.mean([1.896, 1.938])

# Calculate m0 for Landolt stars
m0_landolt = calc_m0(mr_landolt, Is_landolt, Ib_landolt, t_landolt, A_landolt)
print("Calculated m0 for Landolt stars:", m0_landolt)
print("Mean m0 for Landolt stars:", np.mean(m0_landolt))
print("Difference from mean m0 for Landolt stars:", m0_landolt - np.mean(m0_landolt))

# Calculate mr for SN
mr_SN = calc_mr(np.mean(m0_landolt), 6266, 1899.4044617835425, 900, np.nanmedian([1.358,1.255,1.243]))
print("Calculated mr for SN:", mr_SN)


