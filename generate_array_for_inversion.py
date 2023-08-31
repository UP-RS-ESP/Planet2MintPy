#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Bodo Bookhagen and Ariane Mueting

"""
import numpy as np
import os, argparse, glob, tqdm, gzip
from scipy.stats import circstd
from skimage import measure
from skimage.morphology import closing, disk
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import correlation_confidence as cc
from osgeo import gdal

DESCRIPTION = """
Create binary mask of landslide bodies based on the standard deviation of movement direction.
"""

EXAMPLE = """example:
generate_landslide_mask.py \
    --offset_tif_fn "disparity_maps/*_polyfit-F.tif" \
    --area_name aoi3 \
    --npy_out_path masks2 \
    --threshold_angle 45 \
    --threshold_size 5000 \
    --out_pngfname aoi3_landslide_mask.png
"""

def cmdLineParser():
    from argparse import RawTextHelpFormatter
    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--offset_tif_fn', help='2 Band offset file containing dx and dy data. Make sure to put into "quotes" when using wildcards (e.g., *).', required=True)
    parser.add_argument('--npy_out_path', help='Output compressed numpy files', required=True)
    parser.add_argument('--area_name', help='Name of area of interest', required=True)
    parser.add_argument('--out_pngfname', default='', help='Output PNG showing directional standard deviations, mask, and labels', required=False)
    parser.add_argument('-ta', '--threshold_angle', type=np.int8, default=45, help='Threshold of direction standard deviation in degrees', required=False)
    parser.add_argument('-ts', '--threshold_size', type=np.int16, default=5000, help='Threshold of connected pixels for a region to be considered a landslide', required=False)

    return parser.parse_args()


if __name__ == '__main__':
    args = cmdLineParser()

    # #Debugging:
    # #testing purposes:
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    # args.offset_tif_fn = "disparity_maps/*_polyfit-F.tif"
    args.area_name = "aoi3"
    args.npy_out_path = 'npy2'
    # args.threshold_angle = 45
    # args.threshold_size = 5000
    area_name = os.path.join(args.npy_out_path, args.area_name)

    directions_sd_mask_npy_fname = area_name + '_directions_sd_mask.npy.gz'
    directions_sd_mask_geotiff_fname = area_name + '_directions_sd_mask.tif'
    date0_stack_fname = area_name + "_date0.npy.gz"
    date1_stack_fname = area_name + "_date1.npy.gz"
    deltay_stack_fname = area_name + "_deltay.npy.gz"
    dx_npy_fname = area_name + "_dx.npy.gz"
    dy_npy_fname = area_name + "_dy.npy.gz"
    ts_dangle_npy_fname = area_name + "_ts_dangle.npy.gz"

    #Load masked file - either as Geotiff or as npy
    print('Load mask data')
    if os.path.exists(directions_sd_mask_geotiff_fname):
        ds = gdal.Open(directions_sd_mask_geotiff_fname)
        dxdy_size = ds.GetRasterBand(1).ReadAsArray().shape
        mask = ds.GetRasterBand(1).ReadAsArray()
        mask[mask == -9999] = np.nan
        gt = ds.GetGeoTransform()
        sr = ds.GetProjection()
        ds = None
    elif os.path.exists(directions_sd_mask_npy_fname):
        f = gzip.GzipFile(directions_sd_mask_npy_fname, "r")
        mask = np.load(f)
        f = None

    ### Load time series data stored in npy files
    f = gzip.GzipFile(date0_stack_fname, "r")
    date0_stack = np.load(f)
    f = None

    f = gzip.GzipFile(date1_stack_fname, "r")
    date1_stack = np.load(f)
    f = None

    f = gzip.GzipFile(deltay_stack_fname, "r")
    deltay_stack = np.load(f)
    f = None

    print('Load dx data')
    f = gzip.GzipFile(dx_npy_fname, "r")
    dx_stack = np.load(f)
    f = None

    print('Load weight or confidence data')
    f = gzip.GzipFile(ts_dangle_npy_fname, "r")
    conf = np.load(f)
    f = None

    # Extract values only for masked areas
    idxxy = np.where(mask.ravel() == 1)[0]
    nre = int(len(idxxy))
    dx_stack_masked = np.empty((dx_stack.shape[0], nre), dtype=np.float32)
    dx_stack_masked.fill(np.nan)
    conf_masked = np.empty((dx_stack.shape[0], nre), dtype=np.float32)
    conf_masked.fill(np.nan)

    # Could also do this via numba, but looks fast enough right now
    for i in tqdm.tqdm(range(dx_stack.shape[0])):
        dx_stack_masked[i,:] = dx_stack[i, :, :].ravel()[idxxy]
        conf_masked[i,:] = conf[i, :, :].ravel()[idxxy]


    
