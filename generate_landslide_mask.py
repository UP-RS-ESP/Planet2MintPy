#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ariane Mueting and Bodo Bookhagen

"""
import numpy as np
import os
import argparse
import glob
import rasterio
from scipy.stats import circstd
from skimage import measure
from skimage.morphology import closing, disk
import gzip 

DESCRIPTION = """
Create binary mask of landslide bodies based on the standard deviation of movement direction.
"""

def cmdLineParser():
    from argparse import RawTextHelpFormatter
    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--offset_tif_fn', help='2 Band offset file containing dx and dy data. Make sure to put into "quotes" when using wildcards (e.g., *).', required=True)
    parser.add_argument('--npy_out_path', help='Output compressed numpy files', required=True)
    parser.add_argument('--area_name', help='Name of area of interest', required=True)
    parser.add_argument('-ta', '--threshold_angle', type=np.int8, default=45, help='Threshold of direction standard deviation in degrees', required=False)
    parser.add_argument('-ts', '--threshold_size', type=np.int8, default=5000, help='Threshold of connected pixels for a region to be considered a landslide', required=False)

    return parser.parse_args()

def read_file(file, b=1):
    with rasterio.open(file) as src:
        return(src.read(b))
    
def calc_direction(fn):
    with rasterio.open(fn) as src:
        # get raster resolution from metadata
        meta = src.meta

        # first band is offset in x direction, second band in y
        dx = src.read(1)
        dy = src.read(2)
        
        if meta["count"] == 3:
           # print("Interpreting the third band as good pixel mask.")
            valid = src.read(3)
            dx[valid == 0] = np.nan
            dy[valid == 0] = np.nan
    
    #calculate angle to north
    north = np.array([0,1])
    #stack x and y offset to have a 3d array with vectors along axis 2
    vector_2 = np.dstack((dx,dy))
    unit_vector_1 = north / np.linalg.norm(north)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2, axis = 2, keepdims = True)
    #there np.tensordot is needed (instead of np.dot) because of the multiple dimensions of the input arrays
    dot_product = np.tensordot(unit_vector_1,unit_vector_2, axes=([0],[2]))

    direction = np.rad2deg(np.arccos(dot_product))
    
    #as always the smallest angle to north is given, values need to be substracted from 360 if x is negative
    subtract = np.zeros(dx.shape)
    subtract[dx<0] = 360
    direction = abs(subtract-direction)
    
    return direction
    
if __name__ == '__main__':
    args = cmdLineParser()

    filelist = glob.glob(args.offset_tif_fn)
    filelist.sort()
    
    #TODO: Can we speed this up with numba?
    directions = [np.deg2rad(calc_direction(fn)) for fn in filelist]
    std_dirs = np.rad2deg(circstd(directions, axis=0, nan_policy="omit"))
    
    dbin = np.where(std_dirs < args.threshold_angle, 1, 0)

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
        
    for region in np.unique(labeled):
        if region != 0:
            mask = np.where(labeled == region, 1, 0)
            out_fn = f"{args.npy_out_path}/{args.area_name}_region{region}.npy.gz"
            f = gzip.GzipFile(out_fn, "w")
            np.save(file=f, arr=mask)
            f.close()
            f = None
