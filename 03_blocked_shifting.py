# -*- coding: utf-8 -*-
"""
Apply blocked shifting to a co-registered, destriped SRTM-C tile using TanDEM-X reference surface

Author: Ben Purinton {purinton@uni-potsdam.de}
"""

# This script attempts to rectify remaining complex biases in the 1 arcsec SRTM-C 
# following co-registration and destriping

# In addition to the SRTM-C and TanDEM-X tile, this script also requires the TanDEM-X
# water indication mask (WAM) raster to remove problematic pixels prior to destripings 

# Input SRTM and TanDEM tiles must be 1 arcsecond unprojected (WGS84), thus
# of size 3601 * 3601 pixels. We remove the last row and last column of pixels 
# and use factors of 3600 to break the tile into equal squares ranging in size
# from 1.35-7.2 km. We add a column and row of NaN values to output a block
# shifted raster of the same original size (3601 * 3601 pixels).

# The script will output the following:
    # Figure displaying the results in map-view
    # Block shifted version of the input SRTM-C tile
    
#%% import modules

import os, sys
import numpy as np
from scipy import stats
from osgeo import gdal, gdalnumeric
import matplotlib.pyplot as plt

# ignore some errors that arise
gdal.UseExceptions()
errors = np.seterr(all="ignore")

#%% VARIABLE NAMES (SET THESE)

# base path
bd = "/path/to/working/directory/"
# co-registered, destriped SRTM tile
srtm = bd + "destripe/S24W066/srtm_1arcsec_S24W066_aspcorr_destripe.tif"
# original TanDEM tile
tdm = bd + "tandems/tandem_1arcsec_S24W066.tif"
# water indication mask (WAM) from TanDEM auxiliary rasters used to threshold out bad pixels
WAM = bd + "tandems/auxiliary/tandem_1arcsec_S24W066_WAM.tif"
# directory to output results based on current tile Lat/Lon
out_dir = bd + "blockshift/S24W066/"
# short name for figures (without spaces), choose something representative of the chosen parameters
shortname = "S24W066_SRTM_blockshift"

# scale factor for generating slope and hillshade from unprojected DEMs in GDAL
scale_factor = 111120 # DO NOT CHANGE
# resolution of SRTM / TanDEM in approximate meters 
resolution_m = 30. # DO NOT CHANGE

# parameters for shifting
# blocks to run ranging in size from 1.35-7.2 km
chunks = [15., 30., 60., 80.] # these are all factors of 3600 (number of rows and columns in original tile minus one)
max_shft = 1 # maximum allowable vertical shift in meters, suggested values 0.5-1 m to allow minimal influence of outliers


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

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    source = https://stackoverflow.com/questions/16856788/slice-2d-array-into-smaller-2d-arrays
    
    """
    h, w = arr.shape
    sliced = (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))
    
    return sliced

def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    source = https://stackoverflow.com/questions/16856788/slice-2d-array-into-smaller-2d-arrays
    """
    n, nrows, ncols = arr.shape
    whole = (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(h, w))
    
    return whole

#%% Run blocked shifting!

# create output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# get no data value from each dataset
ds = gdal.Open(srtm)
ndv_srtm = ds.GetRasterBand(1).GetNoDataValue()
ds = None
ds = gdal.Open(tdm)
ndv_tdm = ds.GetRasterBand(1).GetNoDataValue()
ds = None

# load datasets
t = gdalnumeric.LoadFile(tdm)
t[t==ndv_tdm]=np.nan
# mask out water pixels using TanDEM-X WAM
w = gdalnumeric.LoadFile(WAM)
t[w >= 33] = np.nan
s = gdalnumeric.LoadFile(srtm)
s[s==ndv_srtm]=np.nan
# calculate dh
dh = t-s

# we remove the last row and column of pixels to create even 3601*3601 square for blocking
if dh.shape[0]==3601 and dh.shape[1]==3601:
    print("unprojected shape of raster in pixels:")
    print(dh.shape)
    print("removing a row and column for even factoring")
    dh = dh[0:-1,0:-1]
else:
    print("somethings wrong, the shape of the unprojected rasters\nshould be 3601*3601, but the shape here is:")
    print(dh.shape)
    print("ending script, check the input DEMs")
    sys.exit()

# get xy grid
xgrid, ygrid = getXYgrid(tdm)

# load hillshade
hillshade = tdm.split(".")[0]+"_HS.tif"
if not os.path.exists(hillshade):
    gdal.DEMProcessing(hillshade, tdm, 'hillshade', scale=scale_factor)
hs = gdalnumeric.LoadFile(hillshade)
hs = hs[0:-1,0:-1]
os.remove(hillshade)

# get slope for weighting the boxes
slope = tdm.split(".")[0] + "_SLOPE.tif"
if not os.path.exists(slope):
    gdal.DEMProcessing(slope, tdm, 'slope', scale=scale_factor)
slp = gdalnumeric.LoadFile(slope)
os.remove(slope)
slp[slp<0]=np.nan
slp = slp[0:-1,0:-1]

# number of rows and columns, should be 3600*3600
rows = dh.shape[0]
cols = dh.shape[1]

# loop through all the chunk sizes
for chunk in chunks:
    
    row_chunk=chunk
    col_chunk=chunk
    
    # block the dh and slope
    dhB = blockshaped(dh, int(dh.shape[0]/row_chunk), int(dh.shape[1]/col_chunk))
    slB = blockshaped(slp, int(dh.shape[0]/row_chunk), int(dh.shape[1]/col_chunk))
    blocks = int(dhB.shape[0])
    height = int(dhB.shape[1]*resolution_m)
    width = int(dhB.shape[2]*resolution_m)
    print()
    print("running a round of blocked shifting on SRTM-C")
    print("there are %i blocks with each %i m wide and %i m tall" % (blocks, height, width))
    save_out = out_dir + srtm.split("/")[-1].split(".")[0]+"_blockshift_"+str(int(height))+"m.tif"
    
    if not os.path.exists(save_out):
    
        # median dh in each block
        # empty holder for medians
        medsDH = np.ones((blocks))*np.nan
        dhB_meds = dhB.copy()
        medsSL = np.ones((blocks))*np.nan
        slB_meds = slB.copy()
        # run each block, saving the stats in the appropriate array
        for k in range(0, blocks):
            # pull out a chunk
            dhi = dhB[k]
            dhi = dhi[~np.isnan(dhi)]
            medsDH[k] = np.nanmedian(dhi)
            # replace values in block with median value
            dhB_meds[k,:,:] = medsDH[k]
            # same for slope
            sli = slB[k]
            sli = sli[~np.isnan(sli)]
            medsSL[k] = np.nanmedian(sli)
            slB_meds[k,:,:] = medsSL[k]
            
        # unblock the median grids
        dh_grid_meds=unblockshaped(dhB_meds, int(rows), int(cols))
        sl_grid_meds=unblockshaped(slB_meds, int(rows), int(cols))
        
        # add NaN row and column to make size 3601*3601 again
        dh_grid_meds = np.insert(dh_grid_meds, dh_grid_meds.shape[0], np.nan, axis=0)
        dh_grid_meds = np.insert(dh_grid_meds, dh_grid_meds.shape[1], np.nan, axis=1)
        sl_grid_meds = np.insert(sl_grid_meds, sl_grid_meds.shape[0], np.nan, axis=0)
        sl_grid_meds = np.insert(sl_grid_meds, sl_grid_meds.shape[1], np.nan, axis=1)
         
        # shift the grid blocks by medians weighting by slope and setting limit on shift amount
        weight_gridB = dhB_meds/slB_meds
        weight_grid=unblockshaped(weight_gridB, int(rows), int(cols))
        weight_grid = np.insert(weight_grid, weight_grid.shape[0], np.nan, axis=0)
        weight_grid = np.insert(weight_grid, weight_grid.shape[1], np.nan, axis=1)
        
        dhB_shft = dhB.copy()
        for k in range(0, blocks):
            if abs(weight_gridB[k][0,0]) < max_shft:
                dhB_shft[k] = dhB_shft[k] - weight_gridB[k]
            else:
                if weight_gridB[k][0,0] < 0:
                    dhB_shft[k] = dhB_shft[k] + max_shft
                if weight_gridB[k][0,0] > 0:
                    dhB_shft[k] = dhB_shft[k] - max_shft
                
        dh_shft=unblockshaped(dhB_shft, int(rows), int(cols))
        dh_shft = np.insert(dh_shft, dh_shft.shape[0], np.nan, axis=0)
        dh_shft = np.insert(dh_shft, dh_shft.shape[1], np.nan, axis=1)
    
        # plot showing results
        fig = plt.figure(figsize=(18,6))
        
        ax = plt.subplot(141)
        im2 = plt.imshow(dh, cmap="RdBu", alpha=1, vmin=-5, vmax=5, 
                         extent=[xgrid.min(), xgrid.max(), ygrid.min(), ygrid.max()])
        plt.title("A: Destriped $dh$\n(med = %0.2f, RMSE = %0.2f)"%(np.nanmedian(dh), RMSE(dh)), 
                  fontsize=12, fontweight='bold', loc='left')
        plt.ylabel('Latitude (deg)')
        plt.grid()
        plt.xlabel("Longitude (deg)")
        cbaxes = fig.add_axes([0.15, 0.075, 0.13, 0.02]) 
        cbar = fig.colorbar(mappable=im2, cax=cbaxes, cmap="RdBu", ticks=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                        label="$dh$ (m)\nTanDEM-X$-$SRTM-C", orientation = 'horizontal')
        cbar.ax.set_yticklabels([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
        
        ax = plt.subplot(142)
        ax.imshow(hs, cmap="gray", extent=[xgrid.min(), xgrid.max(), ygrid.min(), ygrid.max()])
        im2 = ax.imshow(dh_grid_meds, cmap="BrBG", alpha=0.8, vmin=-max_shft-1, vmax=max_shft+1, 
                        extent=[xgrid.min(), xgrid.max(), ygrid.min(), ygrid.max()])
        plt.title("B: Blocked Medians", 
                  fontsize=12, fontweight='bold', loc='left')
        plt.grid()
        plt.xlabel("Longitude (deg)")
        cbaxes = fig.add_axes([0.35, 0.075, 0.13, 0.02]) 
        cbar = fig.colorbar(mappable=im2, cax=cbaxes, cmap="RdBu", ticks=[np.linspace(-max_shft-1, max_shft+1, 5)],
                        label="Blocked Median $dh$ (m)", orientation = 'horizontal')
        
        ax = plt.subplot(143)
        ax.imshow(hs, cmap="gray", extent=[xgrid.min(), xgrid.max(), ygrid.min(), ygrid.max()])
        im2 = ax.imshow(weight_grid, cmap="BrBG", alpha=0.8, vmin=-max_shft, vmax=max_shft, 
                        extent=[xgrid.min(), xgrid.max(), ygrid.min(), ygrid.max()])
        plt.title("C: Blocked Medians\nNormed by Slope", 
                  fontsize=12, fontweight='bold', loc='left')
        plt.grid()
        plt.xlabel("Longitude (deg)")
        cbaxes = fig.add_axes([0.55, 0.075, 0.13, 0.02]) 
        cbar = fig.colorbar(mappable=im2, cax=cbaxes, cmap="RdBu", ticks=[np.linspace(-max_shft, max_shft, 5)],
                        label="Blocked Normed Median $dh$ (m)", orientation = 'horizontal')
        
        ax = plt.subplot(144)
        im2 = plt.imshow(dh_shft, cmap="RdBu", alpha=1, vmin=-5, vmax=5, 
                         extent=[xgrid.min(), xgrid.max(), ygrid.min(), ygrid.max()])
        plt.title("D: Block Shifted $dh$\n(med = %0.2f, RMSE = %0.2f)"%(np.nanmedian(dh_shft), RMSE(dh_shft)), 
                  fontsize=12, fontweight='bold', loc='left')
        plt.grid()
        plt.xlabel("Longitude (deg)")
        cbaxes = fig.add_axes([0.75, 0.075, 0.13, 0.02]) 
        cbar = fig.colorbar(mappable=im2, cax=cbaxes, cmap="RdBu", ticks=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                        label="$dh$ (m)\nTanDEM-X$-$SRTM-C", orientation = 'horizontal')
        cbar.ax.set_yticklabels([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
        
        plt.savefig(out_dir + shortname + "_blocked_shifting_"+str(int(height))+"m.png", bbox_inches="tight", dpi=300)
        plt.close()
        
        
        # create new srtm applying the block shifts to it and outputting a GeoTiff
        s_newB = blockshaped(gdalnumeric.LoadFile(srtm)[0:-1,0:-1], int(dh.shape[0]/row_chunk), 
                             int(dh.shape[1]/col_chunk))
        
        # loop through all blocks and shift by weighted amount
        for k in range(0, blocks):
            if abs(weight_gridB[k][0,0]) < max_shft:
                s_newB[k] = s_newB[k] + weight_gridB[k]
            else:
                if weight_gridB[k][0,0] < 0:
                    s_newB[k] = s_newB[k] - max_shft
                if weight_gridB[k][0,0] > 0:
                    s_newB[k] = s_newB[k] + max_shft
        
        # unblock the shifted grid
        s_new=unblockshaped(s_newB, int(rows), int(cols))
        
        # again add row and column of NaN values to make size 3601*3601 pixels
        s_new = np.insert(s_new, s_new.shape[0], np.nan, axis=0)
        s_new = np.insert(s_new, s_new.shape[1], np.nan, axis=1)
        
        # output tiff
        array2rast(s_new, srtm, save_out)
        print("blocked shifting complete on SRTM-C: %s"%save_out)
        
    else:
        print("%s already exists"%save_out)