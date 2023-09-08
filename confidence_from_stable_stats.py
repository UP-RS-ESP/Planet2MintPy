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
    
    
def confidence_from_stable_stats(aoi, stats_df, max_iqr = 0.5, out_path = "./"): 
    
    stats_df["dx_iqr"] = stats_df.dx_p75-stats_df.dx_p25
    stats_df["dy_iqr"] = stats_df.dy_p75-stats_df.dy_p25
    
    stats_df = stats_df.loc[stats_df.dx_iqr <= max_iqr]
    stats_df = stats_df.loc[stats_df.dy_iqr <= max_iqr]

    stats_df["dx_weight"] = 1 / (stats_df.dx_iqr)
    stats_df["dy_weight"] = 1 / (stats_df.dy_iqr)
    
    stats_df.dx_weight = stats_df.dx_weight.map(lambda x: fixed_val_scaler(x, 0, max_iqr))
    stats_df.dy_weight = stats_df.dy_weight.map(lambda x: fixed_val_scaler(x, 0, max_iqr))

    
    
    for idx, row in tqdm(stats_df.iterrows(), total=stats_df.shape[0]):    
    
        ds = gdal.Open(row.file)
        dat = ds.GetRasterBand(1).ReadAsArray()
        ds = None
        
        con_dx = np.zeros(dat.shape)
        con_dy = np.zeros(dat.shape)
    
        con_dx[~np.isnan(dat)] = row.dx_weight
        con_dy[~np.isnan(dat)] = row.dy_weight
                
        if not os.path.isdir(os.path.join(out_path, "confidence")):
            os.makedirs(os.path.join(out_path, "confidence"))
            
        if not os.path.isdir(os.path.join(out_path, "disparity_maps")):
            os.makedirs(os.path.join(out_path, "disparity_maps"))
            
            
        try: 
            fn1 = os.path.join(out_path, "confidence", aoi + "_" + row.date0.replace("-", "") + "_" + row.date1.replace("-", "") + "_confidence_dx.tif")
            fn2 = os.path.join(out_path, "confidence", aoi + "_" + row.date0.replace("-", "") + "_" + row.date1.replace("-", "") + "_confidence_dy.tif")

        except ValueError:
            fn1 = os.path.join(out_path, "confidence", aoi + "_" + datetime.strftime(row.date0, "%Y%m%d") + "_" + datetime.strftime(row.date1, "%Y%m%d") + "_confidence_dx.tif")
            fn2 = os.path.join(out_path, "confidence", aoi + "_" + datetime.strftime(row.date0, "%Y%m%d") + "_" + datetime.strftime(row.date1, "%Y%m%d") + "_confidence_dy.tif")

        cc.write_Geotiff(row.file, con_dx, fn1)
        cc.write_Geotiff(row.file, con_dy, fn2)
        
        bn = os.path.basename(row.file)

        if os.path.isfile(os.path.join(out_path, "disparity_maps", bn)):
            os.remove(os.path.join(out_path, "disparity_maps", bn))
        os.symlink(row.file, os.path.join(out_path, "disparity_maps", bn))

        
file_loc = "/raid-manaslu/amueting/PhD/Project3/PlanetScope_Data/aoi7/*/disparity_maps/*_polyfit-F.tif"
mask_loc = "/raid-manaslu/amueting/PhD/Project3/PlanetScope_Data/aoi7/*/masks/*npy.gz"

#stats_df = get_stable_stats(file_loc, mask_loc)
# link_to = "/raid-manaslu/amueting/PhD/Project3/PlanetScope_Data/aoi7/group1/selected_03/disparity_maps"
# for idx, row in stats_df.iterrows():   
#     iqr_dx = row.dx_p75-row.dx_p25
#     iqr_dy = row.dy_p75-row.dy_p25
    
#     path, cfile = os.path.split(row.file)
    
#     if not os.path.isdir(link_to):
#         os.makedirs(link_to)
        
#     if (iqr_dx <= 0.3) and (iqr_dy <= 0.3):
#         if os.path.isfile(os.path.join(link_to, cfile)):
#             os.remove(os.path.join(link_to, cfile))
#         os.symlink(os.path.join(path, cfile), os.path.join(link_to, cfile))
        
stats_df = pd.read_csv("stable_stats.csv")
confidence_from_stable_stats("aoi7", stats_df, max_iqr = 0.5, out_path= "./PlanetScope_Data/aoi7/oneover_weights")