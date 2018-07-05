# Difference and Measure SRTM-C and TanDEM-X Tiles

https://zenodo.org/badge/latestdoi/139818255

Benjamin Purinton

purinton[at]uni-potsdam.de

20 June 2018

These Python codes are intended for the correction (co-registration, destriping, block shifting) of raw SRTM-C tiles to raw TanDEM-X tiles and also for the mapping of potential vertical land-level changes in conjunction with TanDEM-X auxiliary rasters. The companion paper is:

Purinton, B., and Bookhagen, B.: Measuring Decadal Vertical Land-level Changes from SRTM-C (2000) and TanDEM-X (~2015) in the South-Central Andes, Earth Surface Dynamics, 2018

The scripts assume some knowledge of Python coding and access to the following packages: numpy, scipy, osgeo (gdal + gdalnumeric), matplotlib, scikit-image.


Below a suggested processing routine:


(1) Create a working directory with a folder for SRTM-C tiles ("/path/to/working/directory/srtms/"), TanDEM-X tiles ("/path/to/working/directory/tandems/"), and TanDEM-X auxiliary rasters ("/path/to/working/directory/tandems/auxiliary/") including the water indication mask (WAM), height error mask (HEM) and consistency mask (COM).

(2) Make sure all tiles and auxiliary rasters are in WGS84 unprojected geographic coordinates and are of the same resolution and dimensions and place them in the appropriate working directory outlined in (1).

(3) Run the scripts in order (01-05) setting the variable names appropriately before running each step. Note the instructions in comments for each variable. Variables can be found in each script at line numbers:

	(a) 01_coregistration.py - 43-69
	(b) 02_fft_destriping.py - 33-53
	(c) 03_blocked_shifting.py - 36-61
	(d) 04_change_mapping_channels.py - 42-80
	(e) 05_change_mapping_hillslopes.py - 33-62
