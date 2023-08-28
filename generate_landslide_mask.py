#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ariane Mueting and Bodo Bookhagen

"""
import numpy as np
import os
import argparse
import glob
from scipy.stats import circstd
from skimage import measure
from skimage.morphology import closing, disk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gzip
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
    # parser = argparse.ArgumentParser(description='')
    # args = parser.parse_args()
    # args.offset_tif_fn = "disparity_maps/*_polyfit-F.tif"
    # args.area_name = "aoi3"
    # args.npy_out_path = 'masks'
    # args.threshold_angle = 45
    # args.threshold_size = 5000

    filelist = glob.glob(args.offset_tif_fn)
    filelist.sort()
    #need an input tif to obtain array size - will be faster to allocate memory
    input_tif = filelist[0]
    ds = gdal.Open(input_tif)
    dxdy_size = ds.GetRasterBand(1).ReadAsArray().shape
    ds = None

    #TODO: Loading all tif files into numpy array
    print('Loading offset dx, dy from TIF files and storing to numpy array')
    dx_stack, dy_stack = cc.load_tif_stacks(filelist, dxdy_size, mask=False)
    print('Calculating angles for each timestep and pixel')
    directions = cc.calc_angle_numba(dx_stack, dy_stack) # returns angles in degree
    del dx_stack
    del dy_stack # remove from memory
    # std_dirs = cc.angle_variance(directions) # angle_variance scaled between 0 and 1
    print('Calculating std. dev. of angles through time')
    directions_sd = cc.nanstd_numba(directions)
    dbin = np.where(directions_sd < args.threshold_angle, 1, 0)

    print('Filter masked image')
    labeled = measure.label(dbin, background=0, connectivity=2)
    info = measure.regionprops(labeled)
    # Filter connected components based on size
    filtered_labels = []
    for region in info:
        if region.area > args.threshold_size:
            filtered_labels.append(region.label)

    filtered_mask = np.isin(labeled, filtered_labels)

    #remove holes
    footprint = disk(5)
    closed = closing(filtered_mask, footprint)
    labeled = measure.label(closed, background=0, connectivity=2)

    if not os.path.exists(args.npy_out_path):
        os.makedirs(args.npy_out_path)

    if len(args.out_pngfname) > 0:
        print('Create output PNG')
        fig, ax = plt.subplots(1,3, figsize = (16, 6), dpi=300)
        im0 = ax[0].imshow(directions_sd, cmap='viridis', vmin=0, vmax=90)
        cb0 = plt.colorbar(im0, ax=ax[0], location='bottom', pad=0.1)
        cb0.set_label('Std. Dev. Directions (degree)')
        # ax[0].set_title('Std. Dev. Directions')
        im1 = ax[1].imshow(dbin, cmap='gray_r', vmin=0, vmax=1)
        cb0 = plt.colorbar(im1, ax=ax[1], location='bottom', pad=0.1)
        cb0.set_label('mask')
        # cmap = matplotlib.cm.get_cmap('magma', np.unique(labeled))
        im2 = ax[2].imshow(labeled, cmap='magma_r')
        cb0 = plt.colorbar(im2, ax=ax[2], location='bottom', pad=0.1)
        cb0.set_label('labelled segments')
        fig.suptitle('%s'%args.area_name, fontsize=16)
        fig.tight_layout()
        fig.savefig(args.out_pngfname, dpi=300)

    for region in np.unique(labeled):
        if region != 0:
            mask = np.where(labeled == region, 1, 0)
            out_fn = f"{args.npy_out_path}/{args.area_name}_region{region}.npy.gz"
            f = gzip.GzipFile(out_fn, "w")
            np.save(file=f, arr=mask)
            f.close()
            f = None
