# -*- coding: utf-8 -*-
"""
Map potential change pixels from corrected SRTM-C and TanDEM-X

Author: Ben Purinton {purinton@uni-potsdam.de}
"""

# This script searches for dh values (TanDEM-X - SRTM-C) outside of 
# expected noise for higher slope environments like hillslopes

# Input SRTM and TanDEM tiles must be 1 arcsecond unprojected (WGS84). The SRTM-C
# should have been corrected via co-registration, fft destriping, and blocked shifting

# This script does not require any binary mask as all pixels are considered in change mapping

# The script will output the following:
    # Potential change raster (dh) for the entire area of input DEMs


#%% import modules

import os, itertools, sys
import numpy as np
import skimage.morphology as morph
from scipy import ndimage
import scipy.stats as stats
from osgeo import gdal, gdalnumeric

# ignore some errors that arise
gdal.UseExceptions()
errors = np.seterr(all="ignore")

#%% VARIABLE NAMES (SET THESE)

# base path
bd = "/path/to/working/directory/"
# co-registered, destriped, block shifted SRTM tile
srtm = bd + "blockshift/S24W066/srtm_1arcsec_S24W066_aspcorr_destripe_blockshift_3600m.tif"
# original TanDEM tile
tdm = bd + "tandems/tandem_1arcsec_S24W066.tif"
# water indication mask (WAM) from TanDEM auxiliary rasters used to threshold out bad pixels
WAM = bd + "tandems/auxiliary/tandem_1arcsec_S24W066_WAM.tif"
# height error map (HEM) from TanDEM auxiliary rasters used to bin the dh pixels
HEM = bd + "tandems/auxiliary/tandem_1arcsec_S24W066_HEM.tif"
# consistency mask (COM) from TanDEM auxiliary rasters used to bin the dh pixels
COM = bd + "tandems/auxiliary/tandem_1arcsec_S24W066_COM.tif"
# directory to output results based on current tile Lat/Lon
out_dir = bd + "potential_change/S24W066/"
# short name for figures (without spaces), choose something representative of the chosen parameters
shortname = "TanDEM_minus_SRTMcorrected_S24W066_dh_FullArea"

# scale factor for generating slope from unprojected DEMs in GDAL
scale_factor = 111120 # DO NOT CHANGE

# standard deviations of summed patch cutoff
stds = 1 # this reduces the noise, suggested values are 1-3

# additional parameters
lo_cut = 5 # lower percentile cutoff for identifiying outliers in a given bin
hi_cut = 95 # upper percentile cutoff for identifiying outliers in a given bin


#%% Functions

def array2rast(array, rast_in, rast_out, NDV = -9999, filetype=gdal.GDT_Float32):
    """
    Use GDAL to take an input array and a given raster and output a raster with the
    same spatial referencing
    """
    ds = gdal.Open(rast_in)
    # check the array size is correct for the georeferencing
    if ds.GetRasterBand(1).YSize == array.shape[0] and ds.GetRasterBand(1).XSize == array.shape[1]:
        print("array is the right size")
    else:
        print("array is the wrong size")
        sys.exit()
    
    driver = gdal.GetDriverByName("GTiff")
    driver.Register()
    outRaster = driver.Create(rast_out, ds.GetRasterBand(1).XSize,
                              ds.GetRasterBand(1).YSize, 1, filetype)
    gt = ds.GetGeoTransform()
    cs = ds.GetProjection()
    outRaster.SetGeoTransform(gt)
    outRaster.SetProjection(cs)
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array,0,0)
    outband.SetNoDataValue(NDV)
    outband.FlushCache()
    
    del driver, outRaster, gt, cs, outband, ds
    
def RMSE(x):
	"""
	Take the root mean squared error of given array
	"""
	return np.sqrt(np.nansum(x**2)/x[np.isfinite(x)].size)


#%% Run potential change mapping!

# create the output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

if not os.path.exists(out_dir + shortname + "_potential_change.tif"):
    
    # get no data value from each dataset
    ds = gdal.Open(srtm)
    ndv_srtm = ds.GetRasterBand(1).GetNoDataValue()
    ds = None
    ds = gdal.Open(tdm)
    ndv_tdm = ds.GetRasterBand(1).GetNoDataValue()
    ds = None
    ds = gdal.Open(HEM)
    ndv_hem = ds.GetRasterBand(1).GetNoDataValue()
    ds = None
    
    # take dh
    t = gdalnumeric.LoadFile(tdm)
    t[t==ndv_tdm]=np.nan
    s = gdalnumeric.LoadFile(srtm)
    s[s==ndv_srtm]=np.nan
    dh = t-s
    
    # load TanDEM-X HEM and COM for binning dh pixels
    hem = gdalnumeric.LoadFile(HEM).astype(float)
    hem[hem==ndv_hem] = np.nan
    com = gdalnumeric.LoadFile(COM).astype(float)
    com[com<8] = np.nan # we only consider the COM pixel values of > 8, as lower values are very inconsistent bad pixels
    
    # remove inconsistent pixels using TanDEM-X WAM
    wam = gdalnumeric.LoadFile(WAM).astype(float)
    idx = np.where(wam >= 33)
    dh[idx] = np.nan
    
    #get RMSE on all slopes for LoD
    LoD = RMSE(dh[np.isfinite(dh)])
    print("minimum LoD from RMSE on all terrain is: %0.2f"%LoD)
    
    # take slope
    slope = tdm.split(".")[0] + "_SLOPE.tif"
    if not os.path.exists(slope):
            gdal.DEMProcessing(slope, tdm, 'slope', scale=scale_factor)
    slp = gdalnumeric.LoadFile(slope).astype(float)
    slp[slp < 0] = np.nan
    os.remove(slope)
    
    # create slope bins using quantiles to evenly space the data
    qt = np.linspace(0, 1.0, num=5, endpoint=True)
    slp_bins = stats.mstats.mquantiles(slp[np.isfinite(slp)], qt)
    # replace first and last value with minimum and maximum  minus and plus a bit
    slp_bins[0] = np.nanmin(slp) - 0.00001
    slp_bins[-1] = np.nanmax(slp) + 0.00001
    # bin the map
    slp_binned = np.searchsorted(slp_bins, slp)
    slp_binned -= 1 # subtract one for later index calling on the bins
    
    # also bin by height error using quantiles to evenly space the data
    hem_bins = stats.mstats.mquantiles(hem[np.isfinite(hem)], qt)
    hem_bins[0] = np.nanmin(hem) - 0.00001
    hem_bins[-1] = np.nanmax(hem) + 0.00001
    hem_binned = np.searchsorted(hem_bins, hem)
    hem_binned -= 1
    
    # finally bin by consistency mask, using only values of 8, 9, and 10 (see TanDEM-X documentation for meaning)
    com_bins = np.array([7.5, 8.5, 9.5, 10.5])
    com_binned = np.searchsorted(com_bins, com)
    com_binned -= 1
    
    # create bins from all possible combinations of height error, consistency, and relief
    vals = list(itertools.product(np.unique(slp_binned[np.isfinite(slp_binned)]),
                                  np.unique(hem_binned[np.isfinite(hem_binned)]),
                                  np.unique(com_binned[np.isfinite(com_binned)])))
    
    # loop through all possible values taking 5th and 95th percentiles as cutoff values for new dh map
    dh_cut = dh.copy()
    for s, h, c in vals:
        print("slope, height error, consistency bins:\n%0.2f, %0.2f, %i"
              % (slp_bins[s], hem_bins[h], com_bins[c]))
        dh_ = dh.copy()
        dh_[slp_binned != s] = np.nan
        dh_[hem_binned != h] = np.nan
        dh_[com_binned != c] = np.nan
        print("number of pixels in this bin: %i" % len(dh_[np.isfinite(dh_)]))
        lo = np.nanpercentile(dh_, lo_cut)
        hi = np.nanpercentile(dh_, hi_cut)
        print("low cut: %0.1f, hi cut: %0.1f" % (lo, hi))
        print()
        idx = np.where(np.logical_and(dh_ > lo, dh_ < hi))
        dh_cut[idx] = np.nan
    
    # also cutoff values that are well within expected noise of RMSE
    dh_cut[abs(dh_cut) < LoD] = np.nan
    
    # binary opening to remove some values
    dh_b = dh_cut.copy()
    dh_b[np.isnan(dh_b)] = 0
    dh_b[dh_b != 0] = 1
    # use disk shaped structuing element of radius one
    selem = morph.disk(1)
    # opening is erosion then dilation, removes unconnected 1s, connects 0s
    dh_b_morph = morph.binary_opening(dh_b, selem)
    dh_cut[dh_b_morph==0] = np.nan
    
    # add up dh in labeled regions
    # only take consistent positive / negative patches outside of standard deviation threshold
    labels = morph.label(dh_b_morph, connectivity=2)
    dh_foo = dh_cut.copy()
    dh_foo[np.isnan(dh)]=0
    label_sum = ndimage.sum(dh_foo, labels=labels, index=np.unique(labels))
    # assign each label and value
    mydict = (list(enumerate(label_sum)))
    dh_summed = np.vectorize(mydict.__getitem__)(labels)[1]
    # threshold out small patches (less than pre-defined standard deviations)
    idx = np.where(abs(dh_summed) < stds*np.nanstd(label_sum))
    dh_cut[idx] = np.nan
    
    # output potential change raster
    array2rast(dh_cut, srtm, out_dir + shortname + "_potential_change.tif")
    
else:
    print("already output potential change map: %s"%(out_dir + shortname + "_potential_change.tif"))