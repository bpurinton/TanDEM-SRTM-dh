# -*- coding: utf-8 -*-
"""
Destripe a co-registered SRTM-C tile using TanDEM-X reference surface

Author: Ben Purinton {purinton@uni-potsdam.de}
"""

# Destripe the SRTM-C with FFTs using TanDEM-X as a reference surface.

# In addition to the SRTM-C and TanDEM-X tile, this script also requires the TanDEM-X
# water indication mask (WAM) raster to remove problematic pixels prior to destriping

# The script will output the following:
    # Figures demonstrating the results of each iterative step
    # Destriped version of the SRTM-C

# Inspiration was taken from Yamazaki et al. (2017)-GRL: 
    # Yamazaki, D., Ikeshima, D., Tawatari, R., Yamaguchi, T., O’Loughlin, F., Neal, J. C., Sampson, C. C., Kanae, S., and Bates, P. D.: A high
    # accuracy map of global terrain elevations, Geophysical Research Letters, 44, 5844–5853, https://doi.org/10.1002/2017GL072874, 2017.
    
#%% import modules

import os, sys
import numpy as np
from scipy import ndimage
from osgeo import gdal, gdalnumeric
import matplotlib.pyplot as plt

# ignore some errors that arise
gdal.UseExceptions()
errors = np.seterr(all="ignore")

#%% VARIABLE NAMES (SET THESE)

# base path
bd = "/path/to/working/directory/"
# co-registered SRTM tile
srtm = bd + "coreg/S24W066/srtm_1arcsec_S24W066_aspcorr.tif"
# original TanDEM tile
tdm = bd + "tandems/tandem_1arcsec_S24W066.tif"
# water indication mask (WAM) from TanDEM auxiliary rasters used to threshold out bad pixels
WAM = bd + "tandems/auxiliary/tandem_1arcsec_S24W066_WAM.tif"
# directory to output results based on current tile Lat/Lon
out_dir = bd + "destripe/S24W066/"
# short name for figures (without spaces), choose something representative of the chosen parameters
shortname = "S24W066_SRTM_destripe"

# parameters for destriping   
filter_sz = 5 # mean filter for passing over Power Spectral Density, can be varied between 3 and 7
rmse_th = 0.05 # tolerance threshold for RMSE convergence (normal=0.05, aggressive=0.02)
percentile_th = 97.5 # percentile threshold for noise cutoff in PSD (normal=97.5, aggressive=95)


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

def getXYgrid(dem):
    """
    takes input geo raster and outputs numpy arrays of X and Y coordinates (center of pixel) 
    """
    # create X and Y
    ds = gdal.Open(dem)
    s = gdalnumeric.LoadFile(dem)
    cols = s.shape[1]
    rows = s.shape[0]
    gt = ds.GetGeoTransform()
    ds = None
    # size of grid (minx, stepx, 0, maxy, 0, -stepy)
    minx, maxy = gt[0], gt[3]
    maxx, miny = gt[0] + gt[1] * cols, gt[3] + gt[5] * rows  
    step = gt[1]
    # center of pixel
    ygrid = np.arange(miny + (step / 2), maxy, step) 
    xgrid = np.arange(minx + (step / 2), maxx, step)
    xgrid, ygrid = np.meshgrid(xgrid, ygrid)
    ygrid = np.flipud(ygrid)
    
    return xgrid, ygrid

#%% Run destriping!

# create directories for output destriped SRTM-C and figures
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# name for destriped SRTM-C
save_out = out_dir+srtm.split("/")[-1].split(".")[0]+"_destripe.tif"

# run destriping if it hasn't already been done
if not os.path.exists(save_out):
    print("running %s"%shortname)
    
    # dummy RMSE variables for first two runs
    RMSEs = [20000, 15000]
    
    # iteration counter
    iteration = 1
    
    # instantiate empty SRTM-C new grid variable
    s_new = []
    
    # get no data value from each dataset
    ds = gdal.Open(srtm)
    ndv_srtm = ds.GetRasterBand(1).GetNoDataValue()
    ds = None
    ds = gdal.Open(tdm)
    ndv_tdm = ds.GetRasterBand(1).GetNoDataValue()
    ds = None
    
    # stop when new RMSE is < X% improvement
    while abs(RMSEs[iteration]-RMSEs[iteration-1]) > RMSEs[iteration-1]*rmse_th:
        
        print("iteration: %i"%iteration)
        
        # for first iteration we use the original SRTM
        if iteration==1:
            s = gdalnumeric.LoadFile(srtm)
            s[s==ndv_srtm] = np.nan
        # subsequent iterations use the destriped SRTM from the previous step
        else:
            s = s_new.copy()
            
        # load the tandem
        t = gdalnumeric.LoadFile(tdm)
        t[t==ndv_tdm] = np.nan
        
        # mask water pixels from TanDEM-X WAM
        w = gdalnumeric.LoadFile(WAM)
        t[w >= 33] = np.nan
        
        # calculate dh
        anom_ref = t-s
        
        # immediately break if original RMSE is above 10, destriping won't work
        if RMSE(anom_ref) > 10:
            print("not running destriping, RMSE too high\nthis tile may be of limited use\nno destriped SRTM-C output")
            sys.exit()
        
        # append original RMSE for first iteration
        if iteration==1:
            RMSEs.append(RMSE(anom_ref))
        else:
            pass
        
        # create X and Y for plotting
        xgrid, ygrid = getXYgrid(tdm)
        
        # now 2D fourier transform to convert to spectral density field
        anom_ref_foo = np.copy(anom_ref)
        # all nans need to be set to 0 elevation difference
        anom_ref_foo[np.isnan(anom_ref)]=0
        # take the fft and also shift to put nyquist in middle
        fft=np.fft.fftshift(np.fft.fft2(anom_ref_foo))
        fft_filt = np.copy(fft)

        # take power spectrum
        psd=abs(fft)**2
        # take mean filter
        psd_mean = ndimage.uniform_filter(psd, size=filter_sz)
        # take ratio
        ratio = psd/psd_mean
        # remove the top X%, given by percentile_th variable
        ratio_th = np.nanpercentile(ratio, percentile_th)
        idx=np.where(ratio<ratio_th)
        fft_filt[idx]=0
        
        # convert to stripes
        stripes = np.real(np.fft.ifft2(np.fft.ifftshift(fft_filt)))
        # add stripes back to SRTM-C to remove
        s_new = s+stripes
        # take new anomaly map
        anom_new = t-s_new
        
        # calculate the new RMSE
        RMSEs.append(RMSE(anom_new))
        
        print("iteration complete, generating figure")
        
        # create a figure to show results
        fig = plt.figure(figsize=(14,6))
        plt.subplot(1,3,1)
        cax = plt.imshow(anom_ref, extent=[xgrid.min(), xgrid.max(), ygrid.min(), ygrid.max()], cmap="RdBu", vmin=-5, vmax=5)
        plt.title("A: Original $dh$\n(med = %0.2f, RMSE = %0.2f)"%(np.nanmedian(anom_ref), RMSE(anom_ref)), 
                  fontsize=12, fontweight='bold', loc='left')
        plt.grid()
        plt.ylabel("Latitude (deg)", fontsize=12)
        
        plt.subplot(1,3,2)
        cax = plt.imshow(stripes, extent=[xgrid.min(), xgrid.max(), ygrid.min(), ygrid.max()], cmap="RdBu", vmin=-5, vmax=5)
        plt.title("B: SRTM-C Stripes from FFT", 
                  fontsize=12, fontweight='bold', loc='left')
        plt.grid()
        plt.xlabel("Longitude (deg)", fontsize=12)
        
        plt.subplot(1,3,3)
        cax = plt.imshow(anom_new, extent=[xgrid.min(), xgrid.max(), ygrid.min(), ygrid.max()], cmap="RdBu", vmin=-5, vmax=5)
        plt.title("C: Destriped $dh$\n(med = %0.2f, RMSE = %0.2f)"%(np.nanmedian(anom_new), RMSE(anom_new)), 
                  fontsize=12, fontweight='bold', loc='left')
        plt.grid()
        
        cbaxes = fig.add_axes([0.21, 0.1, 0.6, 0.04]) 
        cbar = fig.colorbar(mappable=cax, cax=cbaxes, cmap="RdBu", ticks=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                        label="$dh$ (TanDEM-X$-$SRTM-C) or Stripe Magnitude (m)", orientation = 'horizontal')
        cbar.ax.set_yticklabels([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])

        plt.savefig(out_dir + shortname + "_destriping"+str(int(iteration))+".png", bbox_inches="tight", dpi=300)
        plt.close()
               
        # increase counter and break if more than 10 iterations
        iteration += 1
        if iteration==11:
            print("ten iterations reached, breaking script, something's fishy with the tile")
            break
        
    # save out final destriped srtm
    print("RMSE converged, destriping done, outputting new destriped SRTM")
    array2rast(s_new, srtm, save_out)
    print("destriping complete, destriped SRTM-C: %s"%save_out)
    
else:
    print("already ran %s"%shortname)

        
    
