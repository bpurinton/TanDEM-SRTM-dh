# -*- coding: utf-8 -*-
"""
Map potential change pixels from corrected SRTM-C and TanDEM-X

Author: Ben Purinton {purinton@uni-potsdam.de}
"""

# This script searches for dh values (TanDEM-X - SRTM-C) outside of 
# expected noise for low slope environments, such as gravel-bed rivers

# Input SRTM and TanDEM tiles must be 1 arcsecond unprojected (WGS84). The SRTM-C
# should have been corrected via co-registration, fft destriping, and blocked shifting

# This script requires a binary mask of the pixels of interest as input. This mask
# can be generated from a hand-clicked polygon shapefile in a GIS and then rasterized 
# to the resolution of the SRTM / TanDEM (1 arcsec) and output as a GeoTIFF
# The values for the raster should be 1 for inside the AOI and 0 for outside.
# The raster does not need to be the same extents as the SRTM and TanDEM tiles, 
# as the script will generate a reprojected binary mask covering the area and 
# delete this mask when processing is complete.

# RECOMMENDATION: Apply a negative buffer of around -60 m (SRTM resolution limit) 
# to the AOI polygon prior to generation of the binary mask 
# to avoid the inclusion of hillslope pixels with higher uncertainties.

# The script will output the following:
    # Potential change raster (dh) for the area inside the binary mask


#%% import modules

import os, itertools, sys, copy
import numpy as np
import skimage.morphology as morph
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
shortname = "TanDEM_minus_SRTMcorrected_S24W066_dh_Rivers"

# MASK for pixels of interest, can be river pixels, etc.
# Mask should be binary: 0=pixel outside area of interest, 1=pixel inside area of interest
# Binary raster masks can be generated from hand-clicked shapefiles in QGIS or ArcGIS
# Make sure the mask is the same resolution and projection as SRTM / TanDEM tiles
# Mask does not need to cover the same area as the tiles, as it will be reprojected in this script
#mask = bd + "/path/to/binary/river_mask_1arcsec.tif"
mask = bd + "masks/binary_river_mask_1arcsec.tif"

# scale factor for generating slope from unprojected DEMs in GDAL
scale_factor = 111120 # DO NOT CHANGE
# resolution of SRTM / TanDEM in approximate meters 
resolution_m = 30. # DO NOT CHANGE

# additional parameters
RMSE_slp_th = 5 # threshold of slope in degrees for selecting RMSE level of detection
relief_radius = 500 # local relief in meters for binning the dh pixels by
lo_cut = 5 # lower percentile cutoff for identifiying outliers in a given bin
hi_cut = 95 # upper percentile cutoff for identifiying outliers in a given bin


#%% Functions

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
    
    # reproject the mask to the tile area
    mask_clip = mask.split(".")[0] + "_clip.tif"
    if not os.path.exists(mask_clip):
        reproj(mask, srtm, mask_clip, NDV=0, filetype=gdal.GDT_Byte, options=["NBITS=1"])
    
    
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
    
    # mask out dh outside area of interest
    dh, stable = Masking(dh, mask_clip)
    
    # load TanDEM-X HEM and COM for binning dh pixels
    hem = gdalnumeric.LoadFile(HEM).astype(float)
    hem[hem==ndv_hem] = np.nan
    com = gdalnumeric.LoadFile(COM).astype(float)
    com[com<8] = np.nan # we only consider the COM pixel values of > 8, as lower values are very inconsistent bad pixels
    
    # remove inconsistent pixels using TanDEM-X WAM
    wam = gdalnumeric.LoadFile(WAM).astype(float)
    idx = np.where(wam >= 33)
    dh[idx] = np.nan
    
    # get RMSE on low slope areas outside channel for LoD
    slope = tdm.split(".")[0] + "_SLOPE.tif"
    if not os.path.exists(slope):
        gdal.DEMProcessing(slope, tdm, 'slope', scale=scale_factor)
    slp = gdalnumeric.LoadFile(slope)
    os.remove(slope)
    slp[slp<0]=np.nan
    stable = stable[slp < RMSE_slp_th]
    LoD = RMSE(stable[np.isfinite(stable)])
    print("minimum LoD from RMSE on stable, low-slope (%0.1f degree) terrain is: %0.2f"%(RMSE_slp_th, LoD))
    
    # take local 500 m relief for binning
    if not os.path.exists(tdm.split(".")[0]+"_"+str(int(relief_radius))+"m_REL.tif"):
        print("%0.1f m relief does not exist, generating now, this might take a few minutes"%relief_radius)
        r = relief_radius # radius of relief
        disk = morph.disk(r/resolution_m)
        rel = morph.dilation(t, disk)-morph.erosion(t, disk)
        array2rast(rel, tdm, tdm.split(".")[0]+"_"+str(int(relief_radius))+"m_REL.tif")
    else:
        rel = gdalnumeric.LoadFile(tdm.split(".")[0]+"_"+str(int(relief_radius))+"m_REL.tif")
    rel, _ = Masking(rel, mask_clip)
    
    # create relief bins every 50 meters
    rel_bins = np.arange(0, np.nanmax(rel) + 50, 50)
    # bin the map
    rel_binned = np.searchsorted(rel_bins, rel)
     # subtract one for later index calling on the bins
    rel_binned -= 1
    
    # also bin by height error using quantiles to evenly space the data
    qt = np.linspace(0, 1.0, num=5, endpoint=True)
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
    vals = list(itertools.product(np.unique(rel_binned[np.isfinite(rel_binned)]),
                                  np.unique(hem_binned[np.isfinite(hem_binned)]),
                                  np.unique(com_binned[np.isfinite(com_binned)])))
    
    # loop through all possible values taking 5th and 95th percentiles as cutoff values for new dh map
    dh_cut = dh.copy()
    for r, h, c in vals:
        print("relief, height error, consistency bins:\n%0.1f, %0.2f, %i"
              % (rel_bins[r], hem_bins[h], com_bins[c]))
        dh_ = dh.copy()
        dh_[rel_binned != r] = np.nan
        dh_[hem_binned != h] = np.nan
        dh_[com_binned != c] = np.nan
        print("number of pixels in this bin: %i" % len(dh_[np.isfinite(dh_)]))
        lo = np.nanpercentile(dh_, lo_cut)
        hi = np.nanpercentile(dh_, hi_cut)
        print("low cut: %0.1f, hi cut: %0.1f" % (lo, hi))
        print()
        idx = np.where(np.logical_and(dh_ > lo, dh_ < hi))
        dh_cut[idx] = np.nan
    
    # also cutoff values that are well within expected noise of RMSE on stable terrain
    dh_cut[abs(dh_cut) < LoD] = np.nan
    
    # output the potential change raster
    array2rast(dh_cut, srtm, out_dir + shortname + "_potential_change.tif")
    
    # remove the reprojected binary mask
    os.remove(mask_clip)
    
else:
    print("already output potential change map: %s"%out_dir + shortname + "_potential_change.tif")