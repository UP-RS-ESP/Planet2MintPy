#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 11:57:12 2023

@author: ariane
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import rasterio

def read_file(file, b=1):
    with rasterio.open(file) as src:
        return(src.read(b))
    
def min_max_scaler(x):
    if len(x)>1:
        return (x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))
    elif len(x) == 1: 
        return np.array([1])
    else: 
        return np.array([])
    

def adjust_to_uint16(array):
    #stretch gray values between 0 and 255
    
    img = array.astype(np.float32)
    img[img == 0] = np.nan
    
    img = min_max_scaler(img)
    img = img * (2**8)
    img[np.isnan(img)] = 0

    return img.astype(np.uint16)


id1 = "20200713_141202_1003"
id2 = "20220215_140822_1_1038"

f = f"/home/ariane/Documents/Project3/PlanetScope_Data/aoi7/group1/{id1}_3B_AnalyticMS_SR_clip_b2.tif"
img1 = cv.imread(f, cv.IMREAD_UNCHANGED)
#img1 = adjust_to_uint16(img1)
f = f"/home/ariane/Documents/Project3/PlanetScope_Data/aoi7/group1/{id2}_3B_AnalyticMS_SR_clip_b2.tif"
img2 = cv.imread(f, cv.IMREAD_UNCHANGED)
#img2 = adjust_to_uint16(img2)

comp = f"/home/ariane/Documents/Project3/PlanetScope_Data/aoi7/group1/disparity_maps/{id1}_{id2}_L3B_polyfit-F.tif"


id1 = "20220707_144112_41_247c"
id2 = "20220717_143827_66_2470"

f = f"/home/ariane/Documents/PlanetScope/Siguas/L3B/{id1}_3B_AnalyticMS_SR_clip_b2.tif"
img1 = cv.imread(f, cv.IMREAD_UNCHANGED)
#img1 = adjust_to_uint16(img1)
f = f"/home/ariane/Documents/PlanetScope/Siguas/L3B/{id2}_3B_AnalyticMS_SR_clip_b2.tif"
img2 = cv.imread(f, cv.IMREAD_UNCHANGED)
#img2 = adjust_to_uint16(img2)

comp = f"/home/ariane/Documents/PlanetScope/Siguas/L3B/stereo/{id1}_{id2}L3B_polyfit-F.tif"


flow = cv.calcOpticalFlowFarneback(img1, img2, None, 0.5, 10, 35, 5, 5, 1.2, 0)

dx = flow[:,:,0]
dy = flow[:,:,1]
dx[img1 == 0] = np.nan
dy[img1 == 0] = np.nan

dx -= np.nanmedian(dx)
dy -= np.nanmedian(dy)

mag = np.sqrt(dx**2+dy**2)

north = np.array([0,1])
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


plt.figure()
plt.imshow(mag, cmap = "Reds", vmin = 0)
plt.figure()
plt.imshow(direction, cmap = "hsv", vmin = 0, vmax = 360)


dxc = read_file(comp, 1)
dyc = read_file(comp, 2)

fig, ax = plt.subplots(2,2, figsize = (11, 10))
p = ax[0,0].imshow(dx, vmin = -2, vmax = 2, cmap = "coolwarm")
ax[0,1].imshow(dy, vmin = -2, vmax = 2, cmap = "coolwarm")
ax[1,0].imshow(dxc, vmin = -2, vmax = 2, cmap = "coolwarm")
ax[1,1].imshow(dyc, vmin = -2, vmax = 2, cmap = "coolwarm")

plt.colorbar(p, ax = ax[0,0])
plt.colorbar(p, ax = ax[0,1])
plt.colorbar(p, ax = ax[1,0])
plt.colorbar(p, ax = ax[1,1])

ax[0,0].set_title("Dx Optical Flow")
ax[0,1].set_title("Dy Optical Flow")
ax[1,0].set_title("Dx Cross-correlation")
ax[1,1].set_title("Dy Cross-correlation")

plt.suptitle(f"{id1} and {id2}", fontsize=14)
plt.tight_layout()

plt.savefig("OF_vs_CC_example2.png", dpi = 300)