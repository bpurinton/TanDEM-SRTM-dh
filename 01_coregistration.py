# -*- coding: utf-8 -*-
"""
Co-register 2 gridded DEMs (SRTM-C and TanDEM-X) for use in differencing

Author: Ben Purinton {purinton@uni-potsdam.de}
"""

# This script takes 2 DEMs and co-registers the  specified shifting DEM to the 
# control DEM. Methods follow those outlined in Nuth and Kääb (2011)-The Cryosphere:
    # Nuth, C. and Kääb, A.: Co-registration and bias corrections of satellite elevation data sets for quantifying glacier thickness change, The
    # Cryosphere, 5, 271–290, http://dx.doi.org/10.5194/tc-5-271-2011, 2011.

# The shifting DEM is typically an unprojected SRTM-C tile and the control is an
# unprojected TanDEM-X tile, both resampled to the same resolution
    
# In addition to the SRTM-C and TanDEM-X tile, this script also requires the TanDEM-X
# water indication mask (WAM) raster to remove problematic pixels prior to destriping

# Since this script is intended for use outside of the cryosphere, where the vast majority
# of pixels should be stable terrain (ice-free), no mask (aside from a slope threshold)
# is applied prior to cosine fitting.

# The script will output the following:
    # Figures demonstrating the results of each step
    # Aspect corrected version of the shifting DEMs
    
# Inspiration in places was taken from David Shean's demcoreg repository also on GitHub:
    # https://github.com/dshean/demcoreg
    
#%% import modules

import os, sys, copy
import numpy as np
import scipy
from scipy import stats
import matplotlib.pyplot as plt
from osgeo import gdal, gdalnumeric

# ignore some errors that arise
gdal.UseExceptions()
errors = np.seterr(all="ignore")

#%% VARIABLE NAMES (SET THESE)

# base path
bd = "/path/to/working/directory/"
# original SRTM-C tile 
srtm = bd + "srtms/srtm_1arcsec_S24W066.tif"
# TanDEM-X tile for co-registration, must be same resolution and size as SRTM tile
tdm = bd + "tandems/tandem_1arcsec_S24W066.tif"
# water indication mask (WAM) from TanDEM auxiliary rasters used to threshold out bad pixels
WAM = bd + "tandems/auxiliary/tandem_1arcsec_S24W066_WAM.tif"
# directory to output results, choose something representative of the chosen tile
results = bd + "coreg/S24W066/"
# short name for figures (without spaces), choose something representative of the chosen tile
shortname = "S24W066_TanDEMcntrl_SRTMshft"

# scale factor to divide the dx and dy shift vectors (meters) by
scale_factor = 111120 # DO NOT CHANGE
# resolution of SRTM / TanDEM in approximate meters 
resolution_m = 30. # DO NOT CHANGE

# set thresholds for fitting (these parameters can be modified, but the values here are reasonable)
dh_th = 100 # any dh values above this values (meters) are removed from analysis
dh_maxmin = 50 # set approximate minimum and maximum dh values for output plots
slp_th = 5   # only slopes (in degrees) above this threshold are used in aspect fitting, suggested 5-10 degree value
pts = 1000000 # number of  points to extract randomly for fitting


#%% Functions used

def binby(x, y, nbins=50):
    """
    Wrapper function for scipy.stats.binned_statistic. Takes input array and bins
    x by y over a give number of bins. Outputs dictionary with 
    {bin center, bin median, bin 25th percentile, bin 75th percentile}
    Easy to modify and add more dictionary items.
    """
    bins = np.linspace(np.floor(np.nanmin(x)), np.ceil(np.nanmax(x)), nbins)
    bin_med, bin_edge, bin_num = stats.binned_statistic(x, y, statistic=lambda y: np.nanmedian(y), bins=bins)
    bin_25p, _, _ = stats.binned_statistic(x, y, statistic=lambda y: np.nanpercentile(y, 25), bins=bins)
    bin_75p, _, _ = stats.binned_statistic(x, y, statistic=lambda y: np.nanpercentile(y, 75), bins=bins)
    # add any other stats here with same form as above
    bin_width = bin_edge[1] - bin_edge[0]
    bin_centers = bin_edge[1:] - bin_width/2
    out = {"bin_centers":bin_centers, "bin_medians":bin_med, "bin_25thp":bin_25p, "bin_75thp":bin_75p}
    
    return out

def Masking(array, mask):
    """
    Can be used to mask any array (e.g., DEM) with a given binary tiff (e.g., snow area, vegetated region).
    Binary tiff values must be 1 (inside masked area) or 0 (outside masked area)
    Outputs:
        arr_out - array outside of the masked area
        arr_in  - array inside the masked area
    """
    # open the channel raster
    m = gdal.Open(mask)
    # read as array
    m_arr = np.array(m.GetRasterBand(1).ReadAsArray()).astype(int)
    # purge gdal objs
    m = None    
    # verify the shape
    if not array.shape == m_arr.shape:
        print("\nRasters are not the same size, masking not performed\n")
        sys.exit(1)
    else:
        print("\nRasters are same size, performing masking\n")
    # do masking        
    arr_in = copy.deepcopy(array).astype(float)
    arr_in[m_arr == 0] = np.nan
    arr_out = copy.deepcopy(array).astype(float)
    arr_out[m_arr == 1] = np.nan
    
    return arr_in, arr_out

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
    
def reproj(rast_in, clip_rast, rast_out, NDV=-9999, filetype=gdal.GDT_Float32, options=["NBITS=1"]):
    """
    Takes bounds of clipping raster and uses it to clip another raster to the same area.
    Rasters should be the same resolution
    Choose filetype (gdal.GDT_CFloat32, gdal.GDT_Byte, gdal.GDT_Int16, gdal.GDT_Int32)
    Options is a list of creation options ("-co" in gdal speak), set to None for no options
    """
    clipper = gdal.Open(clip_rast)
    gt = clipper.GetGeoTransform()
    minx, maxy = gt[0], gt[3]
    maxx, miny = gt[0] + gt[1] * clipper.GetRasterBand(1).XSize, gt[3] + gt[5] * clipper.GetRasterBand(1).YSize  
    step = gt[1]
    cmd = gdal.Warp(rast_out, rast_in, creationOptions=options, dstNodata=NDV, outputBounds=(minx, miny, maxx, maxy), xRes=step, yRes=step, outputType=filetype)
    cmd = None

    del cmd
    
def mad(arr):
    """ 
    Normalized Median Absolute Deviation: a "Robust" version of standard deviation.
    Indices variabililty of the sample.
    https://en.wikipedia.org/wiki/Median_absolute_deviation 
    This is our prefered metric to measure sample spread (cf. Hoehle and Hoehle (2009)-ISPRS)
    """
    med = np.nanmedian(arr)
    return 1.4826 * np.nanmedian(np.abs(arr - med))

def nuth_kaab_EQ3(x, a, b, c):
    """
    Equation 3 from Nuth and Kääb (2011) - The Cryosphere for aspect fitting
    """
    return a * np.cos(np.deg2rad(b - x)) + c 

#%% Run the co-registration!

# create results directory
if not os.path.exists(results):
    print("Results directory doesn't exist, creating it now...")
    os.makedirs(results)

# aspect correction
if os.path.exists(results + srtm.split("/")[-1].split(".")[0] + "_aspcorr.tif"):
    print("\nAspect correction already completed\n")
    aspcorr = results + srtm.split("/")[-1].split(".")[0] + "_aspcorr.tif"
    
else:
    print("\nBeginning cosine aspect/slope correction on uncorrected DEM\n")
    
    
    # prepare auxiliary data
    slope = tdm.split(".")[0] + "_SLOPE.tif"
    aspect = tdm.split(".")[0] + "_ASPECT.tif"
    
    # create slope / aspect from gdal
    if not os.path.exists(slope):
        print("Generating slope map")
        gdal.DEMProcessing(slope, tdm, 'slope', scale=scale_factor)
    if not os.path.exists(aspect):
        print("Generating aspect map")
        gdal.DEMProcessing(aspect, tdm, 'aspect', scale=scale_factor)    
    
    # load slope and aspect rasters
    slp = gdalnumeric.LoadFile(slope).astype(float)
    slp[slp < 0] = np.nan
    asp = gdalnumeric.LoadFile(aspect).astype(float)
    asp[asp < 0] = np.nan
    
    # delete slope and aspect rasters after loaded into python
    os.remove(slope)
    os.remove(aspect)
    
    # load the original dem
    shft = gdalnumeric.LoadFile(srtm).astype(float)
    
    # get no data value from each dataset
    ds = gdal.Open(srtm)
    ndv_shft = ds.GetRasterBand(1).GetNoDataValue()
    ds = None
    ds = gdal.Open(tdm)
    ndv_stay = ds.GetRasterBand(1).GetNoDataValue()
    ds = None
    
    # set no data to nan
    shft[shft == ndv_shft] = np.nan
    
    # counter
    iteration = 1
    
    # repeat the process until threshold minimization is reached, or after 10 iterations maximum
    while not iteration==10:
        
        # Prepare the data
        shft_cp = shft.copy()

        # load data
        stay = gdalnumeric.LoadFile(tdm).astype(float)
        # mask water pixels from TanDEM-X WAM
        w = gdalnumeric.LoadFile(WAM)
        stay[w >= 33] = np.nan
        
        # load slope and aspect
        sl = slp.copy()
        ap = asp.copy()
        
        # remove nan from orig
        idx = np.where(np.logical_or(stay == ndv_stay, np.isnan(shft_cp)))
        stay[idx] = np.nan
        shft_cp[idx] = np.nan
        sl[idx] = np.nan
        ap[idx] = np.nan
        
        # remove low slopes
        idx = np.where(sl < slp_th)
        stay[idx] = np.nan
        shft_cp[idx] = np.nan
        sl[idx] = np.nan
        ap[idx] = np.nan
        
        # calculate dh
        dh = stay-shft_cp
        
        # Do the actual fitting
        # threshold data
        idx = np.where(abs(dh) < dh_th)
        sl_ = sl[idx]
        dh_ = dh[idx]
        ap_ = ap[idx]
        
        # take random points if grid is very large
        if len(dh_) > pts:
            idx_pts = np.random.choice(np.arange(len(dh_)), pts, replace=False)
            dh_ = dh_[idx_pts]
            sl_ = sl_[idx_pts]
            ap_ = ap_[idx_pts]
        else:
            pass
        
        # calculate dh/tan(slp)
        dhtanslp = dh_ / np.tan(np.deg2rad(sl_))
        
        # bin
        B = binby(ap_, dhtanslp)
        
        # initial guess for c (see Nuth and Kääb (2011)-EQ(4))
        c_seed = np.nanmedian(dh_) / np.tan(np.deg2rad(np.nanmedian(sl_)))
        
        # fitting
        fit, pcov = scipy.optimize.curve_fit(nuth_kaab_EQ3, ap_[~np.isnan(ap_)], 
                                             dhtanslp[~np.isnan(ap_)], [0., 0., c_seed])
        perr = np.sqrt(np.diag(pcov))
        print("\ncosine solution (See Nuth and Kääb (2011) - The Cryosphere)\ndh/tan(slope) = %0.3f * cos(%0.3f - aspect) + %0.3f\n1-sigma errors (a, b, c): %0.3f, %0.3f, %0.3f" % (fit[0], fit[1], fit[2], perr[0], perr[1], perr[2]))
        
        # calculate fit for plotting
        a = (np.arange(0, 360))
        f_a = nuth_kaab_EQ3(a, fit[0], fit[1], fit[2])
    
        # make plot showing original data and fit
        plt.figure()
        plt.scatter(ap_, dhtanslp, s=1, c="gray", alpha=0.4, label="raw data", rasterized=True)
        plt.errorbar(B["bin_centers"], B["bin_medians"], yerr=[B["bin_medians"] - B["bin_25thp"], B["bin_75thp"] - B["bin_medians"]], 
                     fmt="b.--", lw=1, label="median $\pm$25-75$^{th}$ perc.")
        plt.plot(a, f_a, "r", label="fit EQ, 1-sigma error (a, b, c): %0.3f, %0.3f, %0.3f" % (perr[0], perr[1], perr[2]))
        plt.xlim(0, 360)
        plt.ylim(-dh_maxmin, dh_maxmin)
        plt.xticks(np.arange(0, 390, 30))
        plt.title(shortname + 
                  "\nFit EQ: y = %0.2f cos(%0.2f - x) + %0.2f" % (fit[0], fit[1], fit[2]), fontsize=10)
        plt.legend()
        plt.axhline(color="k", lw=1)
        plt.grid()
        plt.xlabel("aspect (degree)")
        plt.ylabel("dh/tan(slope) (m)\nControl-Shifted")
        plt.savefig(results + shortname +  "_aspect_correction_curve_fit_iteration" + str(iteration) + ".png")
        plt.close()
        
        
        # break if magnitude of shift less than 0.5 m
        if 1.0 > abs(fit[0]):
            print("Magnitude of shift vector <0.5 m, ending correction")
            break
        
        # also break if the cosine fit is bad
        if 1.0 < abs(perr[1]):
            print("Bad fit of cosine function to aspect, ending correction\nno aspect corrected SRTM-C output")
            sys.exit()
        
        # calculate dx, dy, dz from fit parameters after Nuth and Kääb (2011)
        # +dy=N, +dx=E, -dy=S, -dx=W
        dx = np.cos(np.deg2rad(fit[1])) * fit[0]
        dy = np.sin(np.deg2rad(fit[1])) * fit[0]
        dz = fit[2] * np.tan(np.deg2rad(np.nanmedian(sl_)))
        print("\ndx shift: %0.2f m" % (dx))
        print("dy shift: %0.2f m" % (dy))
        print("dz shift: %0.2f m\n" % (dz))
        
        # here we adjust the DEM to the shift vector. 
        # weight the neighborhood of 2*2 cells by the shift vectors
        # dy is negative here since we are doing matrix operations on a geographically projected raster
        w_00 = (resolution_m-dx)*(resolution_m--dy)
        w_10 = dx*(resolution_m--dy)
        w_01 = (resolution_m-dx)*-dy
        w_11 = dx*dy
        
        # calculalate new heights (bilinear resampling)
        # to handle edge values we must leave out the last row and column from resampling
        # for these edge pixels we use the original values without any resampling
        shft_new = np.ones((shft.shape[0]-1,shft.shape[1]-1))
        for i in range(0, shft_new.shape[0]):
            for k in range(0, shft_new.shape[1]):
                shft_new[i,k] = (w_00*shft[i,k] + w_10*shft[i+1,k] + w_01*shft[i,k+1] + \
                                    w_11*shft[i+1,k+1]) / (w_00+w_10+w_01+w_11)
        # set edge pixels to original value
        shft_new = np.column_stack((shft_new, np.ones(shft_new.shape[0])))
        shft_new = np.row_stack((shft_new, np.ones(shft_new.shape[1])))
        shft_new[:,-1] = shft[:,-1]
        shft_new[-1,:] = shft[-1,:]
        # add the mean bias
        shft_new = shft_new + dz
        
        # calculate new dh
        dh_corr = stay-shft_new
        # threshold data
        idx = np.where(abs(dh_corr) < dh_th)
        sl_ = sl[idx]
        dh_ = dh_corr[idx]
        ap_ = ap[idx]
        
        # take random points if grid is very large
        if len(dh_) > pts:
            idx_pts = np.random.choice(np.arange(len(dh_)), pts, replace=False)
            dh_ = dh_[idx_pts]
            sl_ = sl_[idx_pts]
            ap_ = ap_[idx_pts]
        else:
            pass
        
        # calculate dh/tan(slp)
        dhtanslp = dh_ / np.tan(np.deg2rad(sl_))
        
        # bin
        B = binby(ap_, dhtanslp)
        
        # make a new plot showing the aspect bias after correction
        plt.figure()
        plt.scatter(ap_, dhtanslp, s=1, c="gray", alpha=0.4, label="raw data", rasterized=True)
        plt.errorbar(B["bin_centers"], B["bin_medians"], yerr=[B["bin_medians"] - B["bin_25thp"], B["bin_75thp"] - B["bin_medians"]], 
                     fmt="b.--", lw=1, label="median $\pm$25-75$^{th}$ perc.")
        plt.xlim(0, 360)
        plt.ylim(-dh_maxmin, dh_maxmin)
        plt.xticks(np.arange(0, 390, 30))
        plt.title(shortname + "\n"
                  "dx: %0.2f m, dy: %0.2f m, dz: %0.2f m" % (dx, dy, dz), fontsize=10)
        plt.legend()
        plt.axhline(color="k", lw=1)
        plt.grid()
        plt.xlabel("aspect (degree)")
        plt.ylabel("dh/tan(slope) (m)\nControl-Shifted")
        plt.savefig(results + shortname + "_aspect_correction_shifted_iteration" + str(iteration) + ".png")
        plt.close()
        
        # mapview showing aspect correction
        fig = plt.figure(figsize=(12,8))
        ax = plt.subplot(121)
        im = plt.imshow(stay-shft_cp, cmap="RdBu")
        plt.title("Original", fontsize=12)
        cb = fig.colorbar(im, ax=ax, shrink=0.6, aspect=20, fraction=.05,pad=.1, alpha=1, orientation="horizontal")
        plt.clim(-dh_maxmin, dh_maxmin)
        cb.set_label("dh (m)\nControl-Shifted")
        plt.xlabel("East-West Pixels", fontsize=12)
        plt.ylabel("North-South Pixels", fontsize=12)
        ax = plt.subplot(122)
        im = plt.imshow(stay-shft_new, cmap="RdBu")
        plt.title("After Shift", fontsize=12)
        cb = fig.colorbar(im, ax=ax, shrink=0.6, aspect=20, fraction=.05,pad=.1, alpha=1, orientation="horizontal")
        plt.clim(-dh_maxmin, dh_maxmin)
        cb.set_label("dh (m)\nControl-Shifted")
        plt.xlabel("East-West Pixels", fontsize=12)
        plt.ylabel("North-South Pixels", fontsize=12)
        fig.savefig(results + shortname + "_mapview_aspect_correction_iteration" + str(iteration) + ".png", dpi=300)
        plt.close()
        
        # assign the new dem
        del shft
        shft = shft_new.copy()
        
        # assign loop
        iteration += 1
        
        # break the correction if improvement in NMAD is <5%
        if abs(mad(dh)-mad(dh_corr)) < mad(dh_corr)*0.05:
            print("<5% improvement in NMAD, ending correction")
            break

    print("\naspect corrected, writing out raster:\n%s\n" % (srtm.split("/")[-1].split(".")[0] + "_aspcorr.tif"))
    
    # save out the corrected file
    array2rast(shft, srtm, results + srtm.split("/")[-1].split(".")[0] + "_aspcorr.tif")        