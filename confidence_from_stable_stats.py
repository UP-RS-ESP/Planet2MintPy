#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ariane Mueting and Bodo Bookhagen

"""

import glob
import sys
sys.path.append("/raid-manaslu/amueting/PhD/Project3/Planet2MintPy")

import gzip
import numpy as np
from osgeo import gdal
import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm 
from mintpy.utils import writefile
import correlation_confidence as cc


def fixed_val_scaler(x, xmin, xmax):
    return (x-xmin)/(xmax-xmin)

def get_stable_stats(file_loc, mask_loc):
    
    file_list = glob.glob(file_loc)
    if len(file_list) == 0: 
        print("No disparity maps found. Check the provided file location...")
        return
    mask_list = glob.glob(mask_loc)
    if len(mask_list) == 0: 
        print("No landslide masks found. Check the provided file location...")
        return
    
    ds = gdal.Open(file_list[0])
    dxdy_size = ds.GetRasterBand(1).ReadAsArray().shape
    ds = None
    
    stable_mask = np.zeros(dxdy_size)
    for m in mask_list:
        f = gzip.GzipFile(m, "r")
        mask_data = np.load(f)
        f = None
    
        stable_mask += mask_data
            
    stats = []
        
    for f in tqdm(file_list):
        ds = gdal.Open(f)
        bn = os.path.basename(f)
        date0 = bn.split('_')[0]
        #need to distinguish between PSBSD and PS2 scene IDs
        if len(bn.split('_')[3]) == 8:
            date1 = bn.split('_')[3]
        else:
            date1 = bn.split('_')[4]
            
        dx = ds.GetRasterBand(1).ReadAsArray()
        dy = ds.GetRasterBand(2).ReadAsArray()
        
        ds = None
        
        dx[stable_mask == 1] = np.nan
        dy[stable_mask == 1] = np.nan
        
        stats.append({"file": f, 
                      "date0": datetime.strptime(date0, "%Y%m%d"),
                      "date1": datetime.strptime(date1, "%Y%m%d"),
                      "dx_std":np.nanstd(dx), 
                      "dx_p25":np.nanpercentile(dx, 25),
                      "dx_p75":np.nanpercentile(dx, 75),
                      "dy_std":np.nanstd(dy), 
                      "dy_p25":np.nanpercentile(dy, 25),
                      "dy_p75":np.nanpercentile(dy, 75), 
                      "group": os.path.dirname(f).split("/")[-2]}) # TODO: remove
    
    df = pd.DataFrame(stats)
    
    df.to_csv("stable_stats.csv", index = False)    
    return df
    
    
def confidence_from_stable_stats(aoi, stats_df, iqr_max = 0.5, out_path = "./confidence"): 
    
    for idx, row in stats_df.iterrows():    
    
        ds = gdal.Open(row.file)
        dat = ds.GetRasterBand(1).ReadAsArray()
        ds = None
        
        con_dx = np.zeros(dat.shape)
        con_dy = np.zeros(dat.shape)
        
        con_val_dx = fixed_val_scaler(row.dx_p75-row.dx_p25, 0, iqr_max) 
        con_val_dy = fixed_val_scaler(row.dy_p75-row.dy_p25, 0, iqr_max) 
    
        con_dx[~np.isnan(dat)] = con_val_dx
        con_dy[~np.isnan(dat)] = con_val_dy
                
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
            
            
        try: 
            fn1 = os.path.join(out_path, aoi + "_" + row.date0.replace("-", "") + "_" + row.date1.replace("-", "") + "_confidence_dx.tif")
            fn2 = os.path.join(out_path, aoi + "_" + row.date0.replace("-", "") + "_" + row.date1.replace("-", "") + "_confidence_dy.tif")

        except ValueError:
            fn1 = os.path.join(out_path, aoi + "_" + datetime.strftime(row.date0, "%Y%m%d") + "_" + datetime.strftime(row.date1, "%Y%m%d") + "_confidence_dx.tif")
            fn2 = os.path.join(out_path, aoi + "_" + datetime.strftime(row.date0, "%Y%m%d") + "_" + datetime.strftime(row.date1, "%Y%m%d") + "_confidence_dy.tif")

            
        cc.write_Geotiff(row.file, con_dx, fn1)
        cc.write_Geotiff(row.file, con_dy, fn2)

        
file_loc = "/raid-manaslu/amueting/PhD/Project3/PlanetScope_Data/aoi7/group1/disparity_maps/*_polyfit-F.tif"
mask_loc = "/raid-manaslu/amueting/PhD/Project3/PlanetScope_Data/aoi7/masks/*npy.gz"

stats_df = get_stable_stats(file_loc, mask_loc)

#stats_df = pd.read_csv("stable_stats.csv")
confidence_from_stable_stats("aoi7", stats_df, iqr_max= 0.6, out_path= "./PlanetScope_Data/aoi7/group1/confidence")