#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 14:06:29 2023

@author: ariane
"""

import pandas as pd
import matplotlib.pyplot as plt
import glob
import rasterio
import numpy as np
from tqdm import tqdm
import os
import gzip
from skimage import measure


def read_file(file, b=1):
    with rasterio.open(file) as src:
        return(src.read(b))
def fixed_val_scaler(x, xmin, xmax):
    return (x-xmin)/(xmax-xmin)

#####DATA EXTRACTION###################################################################################################

# file_loc = "/raid-manaslu/amueting/PhD/Project3/PlanetScope_Data/aoi6/*/disparity_maps/*_polyfit-F.tif"
# mask= "/raid-manaslu/amueting/PhD/Project3/PlanetScope_Data/aoi6/masks/aoi6_region1.npy.gz"


# f = gzip.GzipFile(mask, "r")
# mask_data = np.load(f)
# f = None

# labeled = measure.label(mask_data, background=0, connectivity=2)
# info = measure.regionprops(labeled)
# centr = info[0].centroid
# y = int(centr[0])
# x = int(centr[1])
# # y = 641
# # x = 738
# pad = 2
# file_list = glob.glob(file_loc)

# dxs = []
# dys = []
# for f in tqdm(file_list):
#     r = read_file(f, 1)
#     dxs.append(np.nanmean(r[y-pad:y+pad+1, x-pad:x+pad+1]))
#     r = read_file(f, 2)
#     dys.append(np.nanmean(r[y-pad:y+pad+1, x-pad:x+pad+1]))
    
# names = [os.path.basename(f) for f in file_list]

# df = pd.DataFrame({"name:":names, "dx":dxs, "dy": dys})
# df.to_csv(f"dx_dy_x{x}_y{y}.csv", index = False)


##DATA PREP#########################################################################################################

max_iqr = 0.5

#aoi6 example
# df = pd.read_csv("./dx_dy_x1001_y1016.csv")
# st = pd.read_csv("./stable_stats_aoi6.csv")

#aoi7 example
df = pd.read_csv("./dx_dy_x738_y641.csv")
st = pd.read_csv("./stable_stats_aoi7.csv")

st["dx_iqr"] = st.dx_p75 - st.dx_p25
st["dy_iqr"] = st.dy_p75 - st.dy_p25

st["file"] = [os.path.basename(f) for f in st.file]
df = pd.merge(df,st[["file",'dx_iqr','dy_iqr']],left_on='name:', right_on= "file", how='left')
df = df.drop(columns = "file")

df["date0"] = pd.to_datetime([f[0:8] for f in df["name:"]])
df["date1"] = pd.to_datetime([f.split("_")[3] if len(f.split("_")[3]) == 8 else f.split("_")[4] for f in df["name:"]])
df["dt"] = (df.date1 - df.date0).dt.days
#df = df.loc[df.dt >= 365*2]

#remove jun/july
# df = df.loc[(df.date0.dt.month < 6) | (df.date0.dt.month > 7)]
# df = df.loc[(df.date1.dt.month < 6) | (df.date1.dt.month > 7)]
df = df.reset_index(drop = True)



fig, ax = plt.subplots(2, 1, figsize = (12, 10))

p = ax[0].scatter(df.dt, df.dx, c = df.dx_iqr)
ax[0].set_xlabel("Temporal baseline [days]")
ax[0].set_ylabel("Dx [pix]")
plt.colorbar(p, label = "IQR Dx across stable terrain [pix]", ax = ax[0])

p = ax[1].scatter(df.dt, df.dy, c = df.dy_iqr)
ax[1].set_xlabel("Temporal baseline [days]")
ax[1].set_ylabel("Dy [pix]")
plt.colorbar(p, label = "IQR Dy across stable terrain [pix]", ax = ax[1])

ax[0].grid()
ax[1].grid()



df.dx_iqr.loc[df.dx_iqr >= max_iqr] = np.nan
df.dy_iqr.loc[df.dy_iqr >= max_iqr] = np.nan

#scaling weights between 0 and 1
df["wdx"] = df.dx_iqr.map(lambda x: fixed_val_scaler(x, 0, max_iqr))
df["wdy"] = df.dy_iqr.map(lambda x: fixed_val_scaler(x, 0, max_iqr))



df.wdx = 1 / (df.wdx**2)
df.wdy = 1 / (df.wdy**2)


#plot how confidence would look like
fig, ax = plt.subplots(2, 1, figsize = (12, 10))

ax[0].scatter(df.dt, df.dx, c = "lightgray")
p = ax[0].scatter(df.dt, df.dx, c = df.wdx)
ax[0].set_xlabel("Temporal baseline [days]")
ax[0].set_ylabel("Dx [pix]")
plt.colorbar(p, label = "Confidence 1/(w**2)", ax = ax[0])

ax[1].scatter(df.dt, df.dy, c = "lightgray")
p = ax[1].scatter(df.dt, df.dy, c = df.wdy)
ax[1].set_xlabel("Temporal baseline [days]")
ax[1].set_ylabel("Dy [pix]")
plt.colorbar(p, label = "Confidence 1/(w**2)", ax = ax[1])

ax[0].grid()
ax[1].grid()


# plt.figure(figsize = (12, 5))
# plt.scatter(df.dt, df.dy, c = df.date1.dt.month)
# plt.xlabel("Temporal baseline [days]")
# plt.ylabel("Dy [pix]")
# plt.colorbar(label = "Month")

##INVERSION#####################################################################################

df = df.loc[df.dx_iqr <= max_iqr]
df = df.loc[df.dy_iqr <= max_iqr]


df = df.sort_values(by=['date0'])
df = df.reset_index(drop = True)
    
dates = pd.unique(df[["date0", "date1"]].values.ravel())
dates = np.sort(dates)

timesteps = pd.DataFrame({"step": np.arange(0, len(dates)-1), "date0": dates[:-1], "date1":dates[1:]})

design_matrix = np.zeros((len(df), len(dates)-1))

for idx in range(len(df)):
    pair = df.iloc[idx]
    design_matrix[idx,:] = np.where((timesteps.date0 >= pair.date0) & (timesteps.date1 <= pair.date1), 1, 0)

fig, ax = plt.subplots(2, 1, figsize = (12,10))
for i, d in enumerate(["x", "y"]):
    w = df[f"d{d}_iqr"]
    for exp in ["1-w", "1/w", "1/(w)**2", "1/(w)**4"]:
        W = np.diag(eval(exp))
        Aw = np.dot(W,design_matrix)
        Bw = np.dot(np.array(df[f"d{d}"]),W)
    
        X , _, _, _ = np.linalg.lstsq(Aw, Bw, rcond=None)
        ax[i].plot(timesteps.date0, np.cumsum(X))
        ax[i].scatter(timesteps.date0, np.cumsum(X), label = exp)
        
        
    X , _, _, _ = np.linalg.lstsq(design_matrix, np.array(df[f"d{d}"]), rcond = None)
    ax[i].plot(timesteps.date0, np.cumsum(X))
    ax[i].scatter(timesteps.date0, np.cumsum(X), label = "no weight")
    
    ax[i].legend()
    ax[i].set_title(f"Inverted timeseries d{d}")
    ax[i].set_xlabel("Date")
    ax[i].set_ylabel(f"Cumulative displacement d{d} [pix]")


ax[0].grid()
ax[1].grid()


###ALTERNATIVE#TRY##################################################################################
#averages all measurements per timestep - works but oversmoothes timeseries

# timesteps = pd.date_range(start=min(df.date0),end=max(df.date1))

# plt.figure()

# for d in ["x", "y"]:
#     vals = np.zeros((len(timesteps), len(df)))
#     weights = np.zeros(vals.shape)
    
#     vals[:] = np.nan
#     weights[:] = np.nan
#     for i, row in df.iterrows():
        
#         active = (timesteps >= row.date0) & (timesteps <= row.date1)
#         vals[active,i] = row[f"d{d}"] / row["dt"]
#         weights[active, i] = row[f"wd{d}"]
        
    
#     mdx = np.nanmean(vals, axis = 1)
#     sdx = np.nanstd(vals, axis = 1)


#     plt.plot(timesteps, np.cumsum(mdx), label = f"Mean d{d}")
#     plt.fill_between(timesteps, np.cumsum(mdx-sdx), np.cumsum(mdx+sdx), alpha=0.2)

#     mdx = []
#     for i in range(vals.shape[0]):
        
#         v = vals[i,:][~np.isnan(vals[i,:])]
#         w = weights[i,:][~np.isnan(weights[i,:])]
    
#         mdx.append(np.average(v, weights = w))


#     plt.plot(timesteps, np.cumsum(mdx),label = f"Weighted mean d{d}")
# plt.legend()


# rdates = np.repeat(timesteps, vals.shape[1])
# rvals = vals.flatten()
# plt.figure()
# plt.scatter(rdates, rvals, s = 1, c = weights.flatten())
# plt.colorbar()

