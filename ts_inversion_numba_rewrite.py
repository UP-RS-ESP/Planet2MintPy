#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Bodo Bookhagen and Ariane Mueting

"""

#Limit number of processes for lstsq inversion - you can parallelize multiple inversion steps through for loops
import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
import numpy as np
import os, glob, tqdm, gzip
import datetime as dt
import matplotlib.pyplot as plt
from osgeo import gdal
from numba import njit, prange
import pandas as pd



def create_design_matrix(num_ifgram, dates0, dates1):
    # create design matrix for num_ifgram and dates in dates0 (start), dates1 (end)
    # A is design matrix (also called G)
    unique_dates = np.union1d(np.unique(dates0), np.unique(dates1))
    num_date = len(unique_dates)

    tbase = [i.days + i.seconds / (24 * 60 * 60) for i in (unique_dates - unique_dates[0])]
    tbase = np.array(tbase, dtype=np.float32) / 365.25

    date12_list = []
    for i in range(len(dates0)):
        date12_list.append('%s_%s'%(dt.datetime.strftime(dates0[i], "%Y%m%d"), dt.datetime.strftime(dates1[i], "%Y%m%d")))

    A = np.zeros((num_ifgram, num_date), np.float32)

    date_list = list(unique_dates)
    date_list = [dt.datetime.strftime(d, "%Y%m%d") for d in date_list]

    for i in range(num_ifgram):
        ind1, ind2 = (date_list.index(d) for d in date12_list[i].split('_'))
        A[i, ind1] = -1
        A[i, ind2] = 1

    # Remove reference date as it can not be resolved
    ref_date = dt.datetime.strftime(min(dates0),"%Y%m%d")

    ind_r = date_list.index(ref_date)
    A = np.hstack((A[:, 0:ind_r], A[:, (ind_r+1):]))
    return A, ref_date, tbase



#@njit(parallel=True)
def linalg_weighted_numba(A, y, weights, tbase_diff, nre, rcond=1e-5):
    #numba-based inversion with weights
    num_date = A.shape[1] + 1
    ts = np.empty((num_date, nre), dtype=np.float32)
    ts.fill(np.nan)
    residuals = np.empty((A.shape[0], nre), dtype=np.float32)
    residuals.fill(np.nan)
    ranks = np.empty(nre, dtype=np.float32)
    ranks.fill(np.nan)

    #will do pixel-by-pixel inversion, because some pixels may not have data
    W = np.diag(weights).astype(np.float64)
    for i in prange(nre):
        y2 = y[:,i].astype(np.float64)
        if np.any(np.isnan(y2)) or np.any(np.isinf(y2)):
            continue
        Aw = np.dot(W, A.astype(np.float64))
        Bw = np.dot(y2, W)
        X, residual, ranks[i], _ = np.linalg.lstsq(Aw, Bw, rcond=rcond)
        if residual.size > 0:
            residuals[:,i] = residual
        else:
            residuals[:,i] = A.astype(np.float64).dot(X)
        ts_diff = X * tbase_diff[:,0]
        ts[0,:] = np.zeros(nre, dtype=np.float32)
        ts[1:, i] = np.cumsum(ts_diff)
    return ts, residuals, ranks


#@njit(parallel=True)
def SBAS_noweights_numba(A, y, tbase_diff, nre, rcond=1e-5):
    #numba-based inversion with no weights
    num_date = A.shape[1] + 1
    ts = np.empty((num_date, nre), dtype=np.float32)
    ts.fill(np.nan)
    residuals = np.empty((A.shape[0], nre), dtype=np.float32)
    residuals.fill(np.nan)
    ranks = np.empty(nre, dtype=np.float32)
    ranks.fill(np.nan)

    #will do pixel-by-pixel inversion, because some pixels may not have data
    for i in prange(nre):
        y2 = y[:,i].astype(np.float64)
        if np.any(np.isnan(y2)) or np.any(np.isinf(y2)):
            continue
        X, residual, ranks[i], _ = np.linalg.lstsq(A.astype(np.float64), y2, rcond=rcond)
        if residual.size > 0:
            residuals[:,i] = residual
        else:
            residuals[:,i] = A.astype(np.float64).dot(X)
        ts_diff = X * tbase_diff[:,0]
        ts[0,:] = np.zeros(nre, dtype=np.float32)
        ts[1:, i] = np.cumsum(ts_diff)
    return ts, residuals, ranks


#@njit(parallel=True)
def NSBAS_noweights_numba(A, y, tbase_diff, tbase, nre, gamma=1e-4, rcond=1e-5):
    #numba-based inversion with no weights
    num_date = A.shape[1] + 1
    num_im = A.shape[0]
    ts = np.empty((num_date, nre), dtype=np.float32)
    ts.fill(np.nan)
    residuals = np.empty((A.shape[0], nre), dtype=np.float32)
    residuals.fill(np.nan)
    ranks = np.empty(nre, dtype=np.float32)
    ranks.fill(np.nan)
    vconst = np.empty(nre, dtype=np.float32)
    vconst.fill(np.nan)
    vel = np.empty(nre, dtype=np.float32)
    vel.fill(np.nan)

    ### Set matrix of NSBAS part (bottom)
    Gbl = np.tril(np.ones((num_date, num_date-1), dtype=np.float32), k=-1) #lower tri matrix without diag
    Gbr = -np.ones((num_date, 2), dtype=np.float32)
    Gbr[:, 0] = -tbase
    # Gbr[:, 0] = tbase_diff
    Gb = np.concatenate((Gbl, Gbr), axis=1)*gamma
    Gt = np.concatenate((A, np.zeros((num_im, 2), dtype=np.float32)), axis=1)
    Gt = np.concatenate((A, np.ones((num_im, 2), dtype=np.float32)), axis=1)
    Gall = np.float32(np.concatenate((Gt, Gb)))

    #will do pixel-by-pixel inversion, because some pixels may not have data
    for i in prange(nre):
        y2 = np.concatenate((y[:, i], np.zeros((num_date), dtype=np.float32))).transpose()
        if np.any(np.isnan(y2)) or np.any(np.isinf(y2)):
            continue
        X, residual, ranks[i], _ = np.linalg.lstsq(Gall.astype(np.float64), y2, rcond=rcond)
        if residual.size > 0:
            residuals[:,i] = residual
        else:
            residuals[:,i] = A.astype(np.float64).dot(X[1:-1])
        ts_diff = X[:num_date-1] * tbase_diff[:,0] #Incremental displacement (num_date-1, n_pt)
        # ts_diff = X[:num_date-1] * tbase_diff[1:] #Incremental displacement (num_date-1, n_pt)
        vel[i] = X[num_date-1] #Velocity (n_pt)
        vconst[i] = X[num_date] #Constant part of linear velocity (c of vt+c) (n_pt)

        ts[0,:] = np.zeros(nre, dtype=np.float32)
        ts[1:, i] = np.cumsum(ts_diff)
    return ts, residuals, ranks, vel, vconst


def read_file(fn, b=1):
    ds = gdal.Open(fn)
    data = ds.GetRasterBand(b).ReadAsArray()
    ds = None
    return data

def min_max_scaler(x):
    if len(x)>1:
        return (x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))
    elif len(x) == 1:
        return np.array([1])
    else:
        return np.array([])

def get_sun_pos(files):

    import subprocess
    import json

    bns = [os.path.basename(f) for f in files]

    ids1 = [("_").join(x.split("_")[0:3]) if len(x.split("_")[3]) == 8 else ("_").join(x.split("_")[0:4]) for x in bns]
    ids2 = [("_").join(x.split("_")[3:]) if len(x.split("_")[3]) == 8 else ("_").join(x.split("_")[4:]) for x in bns]
    ids2 = [("_").join(x.split("_")[0:3]) if (x.split("_")[3] == "L3B") else ("_").join(x.split("_")[0:4]) for x in ids2]

    ids = np.union1d(np.unique(ids1), np.unique(ids2))

    search = f"planet data filter --string-in id {','.join(ids)} > filter.json"
    result = subprocess.run(search, shell = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.stderr != "":
        print(result.stderr)
    search = "planet data search PSScene --limit 0 --filter filter.json > search.geojson"

    result = subprocess.run(search, shell = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.stderr != "":
        print(result.stderr)

    gj = [json.loads(line) for line in open("search.geojson", "r")]


    sun = pd.DataFrame({"id": [f["id"] for f in gj], "date": [f["id"].split("_")[0] for f in gj], "sun_az": [f["properties"]["sun_azimuth"] for f in gj], "sun_elev": [f["properties"]["sun_elevation"] for f in gj]})
    sun.date = pd.to_datetime(sun.date)
    return sun


if __name__ == '__main__':


    files = glob.glob("/home/ariane/Documents/Project3/PlanetScope_Data/aoi7/*/disparity_maps/*L3B_polyfit-F.tif")
    mask_fn = "/home/ariane/Documents/Project3/PlanetScope_Data/aoi7/masks/aoi7_region1.npy.gz"
    files = glob.glob("/raid/Planet_NWArg/PS2_aoi7/disparity_maps/*L3B_polyfit-F.tif")
    mask_fn = "/raid/Planet_NWArg/PS2_aoi7/masks/aoi7_region1.npy.gz"
    bns = [os.path.basename(f) for f in files]
    dx_stack = np.asarray([read_file(f,1) for f in files])
    dy_stack = np.asarray([read_file(f,2) for f in files])
    dates0 = [dt.datetime.strptime(f[0:8], "%Y%m%d") for f in bns]
    dates1 = [dt.datetime.strptime(f.split("_")[3], "%Y%m%d") if len(f.split("_")[3]) == 8 else dt.datetime.strptime(f.split("_")[4], "%Y%m%d") for f in bns]


    f = gzip.GzipFile(mask_fn, "r")
    mask = np.load(f)
    f = None

    area_name = "aoi7"
    deltay_stack_scale = 2


    png_out_path = "./png"
    if not os.path.exists(png_out_path):
        os.mkdir(png_out_path)


    # Extract values only for masked areas
    print('Extract relevant values and remove full array from memory')
    idxxy = np.where(mask.ravel() == 1)[0]
    num_ifgram = dx_stack.shape[0]
    nre = int(len(idxxy))
    dx_stack_masked = np.empty((num_ifgram, nre), dtype=np.float32)
    dx_stack_masked.fill(np.nan)
    dy_stack_masked = np.empty((num_ifgram, nre), dtype=np.float32)
    dy_stack_masked.fill(np.nan)

    # Could also do this via numba, but looks fast enough right now
    for i in tqdm.tqdm(range(dx_stack.shape[0])):
        dx_stack_masked[i,:] = dx_stack[i, :, :].ravel()[idxxy]
        dy_stack_masked[i,:] = dy_stack[i, :, :].ravel()[idxxy]


    del dx_stack, dy_stack

    dates0 = np.asarray(dates0)
    dates1 = np.asarray(dates1)

    ddates = dates1 - dates0
    ddates_day = np.array([i.days for i in ddates])

    # create design_matrix
    A, ref_date, tbase = create_design_matrix(num_ifgram, dates0, dates1)
    tbase_diff = np.diff(tbase).reshape(-1, 1)
    tbase_diff2 = np.insert(tbase_diff, 0, 0)

    print('Number of correlations: %d'%num_ifgram)
    print('Number of unique Planet scenes: %d'%len(tbase))
    nIslands = np.min(A.shape) - np.linalg.matrix_rank(A)
    print('Number of connected components in network: %d '%nIslands)
    if nIslands > 1:
        print('\tThe network appears to be disconnected and contains island components')


    # SBAS - no weights
    print('\nRun linear SBAS inversion on each pixel with no weights')
    print('\t dx')
    dx_ts_SBAS_noweights_numba, dx_residuals_SBAS_noweights_numba, dx_ranks_SBAS_noweights_numba = SBAS_noweights_numba(A, dx_stack_masked, tbase_diff, nre, rcond=1e-5)
    print('\t dy')
    dy_ts_SBAS_noweights_numba, dy_residuals_SBAS_noweights_numba, dy_ranks_SBAS_noweights_numba = SBAS_noweights_numba(A, dy_stack_masked, tbase_diff, nre, rcond=1e-5)

    # NSBAS - no weights
    print('\nRun linear NSBAS inversion on each pixel with no weights')
    print('\t dx')
    dx_ts_NSBAS_noweights_numba, dx_residuals_NSBAS_noweights_numba, dx_ranks_NSBAS_noweights_numba, dx_ranks_NSBAS_noweights_vel, dx_ranks_NSBAS_noweights_vconst = NSBAS_noweights_numba(A, dx_stack_masked, tbase_diff, tbase, nre, rcond=1e-5)
    print('\t dy')
    dy_ts_NSBAS_noweights_numba, dy_residuals_NSBAS_noweights_numba, dy_ranks_NSBAS_noweights_numba, dx_ranks_NSBAS_noweights_vel, dx_ranks_NSBAS_noweights_vconst = NSBAS_noweights_numba(A, dy_stack_masked, tbase_diff, tbase, nre, rcond=1e-5)

    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    ax[0].plot(np.cumsum(tbase_diff2), np.nanmean(dx_ts_NSBAS_noweights_numba, axis=1), '-', color='darkblue', label='NSBAS')
    ax[0].plot(np.cumsum(tbase_diff2), np.nanmean(dx_ts_SBAS_noweights_numba, axis=1), '-', color='firebrick', label='SBAS')
    ax[0].set_title('Mean dx offset (n=%d)'%nre, fontsize=14)
    ax[0].set_xlabel('Time [y]')
    ax[0].set_ylabel('Cumulative dx offset [pix]')
    ax[0].legend()
    ax[0].grid()
    ax[1].plot(np.cumsum(tbase_diff2), np.nanmean(dy_ts_NSBAS_noweights_numba, axis=1), '-', color='darkblue', label='NSBAS')
    ax[1].plot(np.cumsum(tbase_diff2), np.nanmean(dy_ts_SBAS_noweights_numba, axis=1), '-', color='firebrick', label='SBAS')
    ax[1].set_title('Mean dy offset (n=%d)'%nre, fontsize=14)
    ax[1].set_xlabel('Time [y]')
    ax[1].set_ylabel('Cumulative dy offset [pix]')
    ax[1].legend()
    ax[1].grid()
    fig.tight_layout()
    fig.savefig(os.path.join(png_out_path, '%s_dx_dy_SBAS_NSBAS_inversion.png'%area_name), dpi=300)

    #weighted
    sun = get_sun_pos(files)
    dates_df = pd.DataFrame({'date0': dates0, 'date1': dates1})

    weight_df = pd.merge(dates_df, sun, left_on='date0', right_on='date', how='inner')
    weight_df = pd.merge(weight_df, sun, left_on='date1', right_on='date', how='inner', suffixes = ("_ref", "_sec"))
    weight_df.drop(["date_ref", "date_sec"], inplace = True, axis = 1)

    weights = 1-min_max_scaler(np.array(abs(weight_df.sun_az_ref - weight_df.sun_az_sec)))

    print('Run linear inversion on each pixel with weights')
    print('\t dx')
    dx_ts_weights_numba, dx_residuals_weights_numba, dx_ranks_weights_numba = linalg_weighted_numba(A, dx_stack_masked, weights, tbase_diff, nre, rcond=1e-5)
    print('\t dy')
    dy_ts_weights_numba, dy_residuals_weights_numba, dy_ranks_weights_numba = linalg_weighted_numba(A, dy_stack_masked, weights, tbase_diff, nre, rcond=1e-5)


    # dx and dy time series plot
    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    ax[0].plot(np.cumsum(tbase_diff2), np.nanmean(dx_ts_SBAS_noweights_numba, axis=1), '-', color='darkblue', label='No weights')
    ax[0].plot(np.cumsum(tbase_diff2), np.nanmean(dx_ts_weights_numba, axis=1), '-', color='firebrick', label='Weighted')
    ax[0].set_title('Mean dx offset (n=%d)'%nre, fontsize=14)
    ax[0].set_xlabel('Time [y]')
    ax[0].set_ylabel('Cumulative dx offset [pix]')
    ax[0].legend()
    ax[0].grid()
    ax[1].plot(np.cumsum(tbase_diff2), np.nanmean(dy_ts_SBAS_noweights_numba, axis=1), '-', color='darkblue', label='No weights')
    ax[1].plot(np.cumsum(tbase_diff2), np.nanmean(dy_ts_weights_numba, axis=1), '-', color='firebrick', label='Weighted')
    ax[1].set_title('Mean dy offset (n=%d)'%nre, fontsize=14)
    ax[1].set_xlabel('Time [y]')
    ax[1].set_ylabel('Cumulative dy offset [pix]')
    ax[1].legend()
    ax[1].grid()
    fig.tight_layout()
    fig.savefig(os.path.join(png_out_path, '%s_dx_dy_timeseries_scaled_with_different_weights.png'%area_name), dpi=300)
