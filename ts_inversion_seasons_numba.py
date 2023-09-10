#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Bodo Bookhagen and Ariane Mueting

"""

#Limit number of processes for lstsq inversion - you can parallelize multiple inversion steps through for loops
import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
import numpy as np
import os, argparse, glob, tqdm, gzip
import datetime as dt

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import correlation_confidence as cc
from osgeo import gdal

from numba import njit, prange
from numba_progress import ProgressBar

DESCRIPTION = """
Use NDJFMA and MJJASO as seasons and run inversion separately for each season (also for all dates).

Run time series inversion on offset pixels via numpy and numba. Takes advantage of multiple cores, but requires memory. Very fast, but only useful for limited number of points (up to 1e5) and limited timesteps (up to 100).
This reads in the offset timeseries and a landslide mask file (e.g., created with generate_landslide_mask.py) and an uncertainty offset file (IQR, generated with create_offset_confidence.py and --method 2).
"""

EXAMPLE = """example:
ts_inversion_seasons_numba.py \
    --area_name aoi3 \
    --npy_out_path npy \
    --png_out_path npy
"""

def prepare_design_matrix_input(date0_stack, date1_stack, date_format = "%Y%m%d"):
    unique_date0 = np.unique(date0_stack)
    unique_date1 = np.unique(date1_stack)
    unique_date = np.union1d(unique_date0, unique_date1)
    num_date = len(unique_date)
    date1s = unique_date

    # tbase in the unit of years
    date_list = [str(i) for i in np.int32(unique_date).tolist()]
    dates = np.array([dt.datetime.strptime(i, date_format) for i in date_list])
    tbase = [i.days + i.seconds / (24 * 60 * 60) for i in (dates - dates[0])]
    tbase = np.array(tbase, dtype=np.float32) / 365.25

    refDate = None # datelist1[0].strftime("%Y%m%d")
    date0 = [str(i) for i in np.int32(date0_stack).tolist()]
    date1 = [str(i) for i in np.int32(date1_stack).tolist()]
    date12_list = []
    for i in range(len(date0)):
        date12_list.append('%s_%s'%(date0[i], date1[i]))

    return num_date, tbase, date1s, date_list, date12_list, unique_date

def create_design_matrix(num_ifgram, num_date, tbase, date1s, date_list, refDate):
    # create design matrix
    # A for minimizing the residual of phase
    # B for minimizing the residual of phase velocity (not used here)
    A = np.zeros((num_ifgram, num_date), np.float32)
    B = np.zeros((num_ifgram, num_date), np.float32)
    for i in range(num_ifgram):
        ind1, ind2 = (date_list.index(d) for d in date12_list[i].split('_'))
        A[i, ind1] = -1
        A[i, ind2] = 1
        # B[i, ind2:ind1] = tbase[ind2:ind1] - tbase[ind2 + 1:ind1 + 1]
        B[i, ind1] = tbase[ind2] - tbase[ind1]

    # Remove reference date as it can not be resolved
    if refDate != 'no':
        # default refDate
        if refDate is None:
            # for single   reference network, use the same reference date
            # for multiple reference network, use the first date
            if len(set(date1s)) == 1:
                refDate = date1s[0]
            else:
                refDate = date_list[0]

        # apply refDate
        if refDate:
            ind_r = date_list.index(refDate)
            A = np.hstack((A[:, 0:ind_r], A[:, (ind_r+1):]))
            B = B[:, :-1]
    return A, B, refDate


@njit(parallel=True)
def linalg_rweights_numba(A, y, weights, tbase_diff, num_pixel, rcond=1e-5):
    #numba-based inversion using inverse IQR
    # W = (1/W)
    num_date = A.shape[1] + 1
    ts = np.empty((num_date, nre), dtype=np.float32)
    ts.fill(np.nan)
    residuals = np.empty((A.shape[0], nre), dtype=np.float32)
    residuals.fill(np.nan)
    ranks = np.empty(nre, dtype=np.float32)
    ranks.fill(np.nan)

    #will do pixel-by-pixel inversion, because some pixels may not have data
    # print('Run linear inversion on each pixel with 1 / weights')
    for i in prange(nre):
        W = weights[:,i]
        if np.any(np.isnan(W)) or np.any(np.isinf(W)):
            continue
        W = 1/W
        W = np.diag(W).astype(np.float64)
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


@njit(parallel=True)
def linalg_rweights2_numba(A, y, weights, tbase_diff, num_pixel, rcond=1e-5):
    #numba-based inversion using inverse IQR
    # W = (1/(W**2))
    num_date = A.shape[1] + 1
    ts = np.empty((num_date, nre), dtype=np.float32)
    ts.fill(np.nan)
    residuals = np.empty((A.shape[0], nre), dtype=np.float32)
    residuals.fill(np.nan)
    ranks = np.empty(nre, dtype=np.float32)
    ranks.fill(np.nan)

    #will do pixel-by-pixel inversion, because some pixels may not have data
    # print('Run linear inversion on each pixel with 1 / weights')
    for i in prange(nre):
        W = weights[:,i]
        if np.any(np.isnan(W)) or np.any(np.isinf(W)):
            continue
        W = 1/(W**2)
        W = np.diag(W).astype(np.float64)
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


@njit(parallel=True)
def linalg_noweights_numba(A, y, tbase_diff, num_pixel, rcond=1e-5):
    #numba-based inversion with no weights
    num_date = A.shape[1] + 1
    ts = np.empty((num_date, nre), dtype=np.float32)
    ts.fill(np.nan)
    residuals = np.empty((A.shape[0], nre), dtype=np.float32)
    residuals.fill(np.nan)
    ranks = np.empty(nre, dtype=np.float32)
    ranks.fill(np.nan)

    #will do pixel-by-pixel inversion, because some pixels may not have data
    # print('Run linear inversion on each pixel with 1 / weights')
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


@njit(parallel=True)
def linalg_tweights_numba(A, y, weights, deltay_stack2, deltay_stack_scale, tbase_diff, num_pixel, rcond=1e-5):
    #numba-based inversion using inverse IQR and time difference (the longer the duration, the higher the weight)
    # W = (1/W) * deltay_stack * deltay_stack_scale
    num_date = A.shape[1] + 1
    ts = np.empty((num_date, nre), dtype=np.float32)
    ts.fill(np.nan)
    residuals = np.empty((A.shape[0], nre), dtype=np.float32)
    residuals.fill(np.nan)
    ranks = np.empty(nre, dtype=np.float32)
    ranks.fill(np.nan)

    #will do pixel-by-pixel inversion, because some pixels may not have data
    # print('Run linear inversion on each pixel with 1 / weights')
    for i in prange(nre):
        W = weights[:,i]
        if np.any(np.isnan(W)) or np.any(np.isinf(W)):
            continue
        W = (1/W) * deltay_stack2**deltay_stack_scale
        W = np.diag(W).astype(np.float64)
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


def cmdLineParser():
    from argparse import RawTextHelpFormatter
    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--npy_out_path', default='npy', help='Output compressed numpy files', required=True)
    parser.add_argument('--area_name', help='Name of area of interest', required=True)
    parser.add_argument('--png_out_path', default='npy', help='Output PNG showing directional standard deviations, mask, and labels', required=False)
    parser.add_argument('--deltay_stack_scale', default=2., help='Output PNG showing directional standard deviations, mask, and labels', required=False)

    return parser.parse_args()


if __name__ == '__main__':
    args = cmdLineParser()

    # # Debugging:
    # parser = argparse.ArgumentParser(description='')
    # args = parser.parse_args()
    # args.area_name = "aoi7"
    # args.npy_out_path = 'npy'
    # args.png_out_path = 'npy'
    # args.deltay_stack_scale = 2.

    area_name = os.path.join(args.npy_out_path, args.area_name)
    deltay_stack_scale = args.deltay_stack_scale

    if not os.path.exists(args.png_out_path):
        os.mkdir(args.png_out_path)

    directions_sd_mask_npy_fname = area_name + '_directions_sd_mask.npy.gz'
    directions_sd_mask_geotiff_fname = area_name + '_directions_sd_mask.tif'
    date0_stack_fname = area_name + "_date0.npy.gz"
    date1_stack_fname = area_name + "_date1.npy.gz"
    deltay_stack_fname = area_name + "_deltay.npy.gz"
    dx_npy_fname = area_name + "_dx.npy.gz"
    dy_npy_fname = area_name + "_dy.npy.gz"
    ts_dangle_npy_fname = area_name + "_ts_dangle.npy.gz"
    dx_stack_iqr_fn = area_name + '_dx_iqr.npy.gz'
    dy_stack_iqr_fn = area_name + '_dy_iqr.npy.gz'
    dx_ts_NDJFMA_tweights_numba_fn = area_name + '%s_dx_ts_NDJFMA_tweights.npy.gz'%args.area_name
    dy_ts_NDJFMA_tweights_numba_fn = area_name + '%s_dy_ts_NDJFMA_tweights.npy.gz'%args.area_name
    dx_ts_MJJASO_tweights_numba_fn = area_name + '%s_dx_ts_MJJASO_tweights.npy.gz'%args.area_name
    dy_ts_MJJASO_tweights_numba_fn = area_name + '%s_dy_ts_MJJASO_tweights.npy.gz'%args.area_name
    dx_ts_tweights_numba_fn = area_name + '%s_dx_ts_tweights.npy.gz'%args.area_name
    dy_ts_tweights_numba_fn = area_name + '%s_dy_ts_tweights.npy.gz'%args.area_name
    # dx_ts_rweights2_numba_fn = area_name + '%s_dx_ts_rweights2.npy.gz'%args.area_name
    # dy_ts_rweights2_numba_fn = area_name + '%s_dy_ts_rweights2.npy.gz'%args.area_name
    #ts should be in same order as other data
    #ts_confidence_npy_fname = "./confidence_stable_stats/" area_name + "_ts_dangle.npy.gz"
    #aoi3_20190807_20200517_confidence_dy.tif

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

    print('Load dx iqr data')
    f = gzip.GzipFile(dx_stack_iqr_fn, "r")
    dx_iqr = np.load(f)
    f = None

    print('Load dy data')
    f = gzip.GzipFile(dy_npy_fname, "r")
    dy_stack = np.load(f)
    f = None

    print('Load dy iqr data')
    f = gzip.GzipFile(dy_stack_iqr_fn, "r")
    dy_iqr = np.load(f)
    f = None

    # Extract values only for masked areas
    print('Extract relevant values and remove full array from memory')
    idxxy = np.where(mask.ravel() == 1)[0]
    num_ifgram = dx_stack.shape[0]
    nre = int(len(idxxy))
    dx_stack_masked = np.empty((num_ifgram, nre), dtype=np.float32)
    dx_stack_masked.fill(np.nan)
    dy_stack_masked = np.empty((num_ifgram, nre), dtype=np.float32)
    dy_stack_masked.fill(np.nan)
    dx_IQR_masked = np.empty((num_ifgram, nre), dtype=np.float32)
    dx_IQR_masked.fill(np.nan)
    dy_IQR_masked = np.empty((num_ifgram, nre), dtype=np.float32)
    dy_IQR_masked.fill(np.nan)
    # Could also do this via numba, but looks fast enough right now
    for i in tqdm.tqdm(range(dx_stack.shape[0])):
        dx_stack_masked[i,:] = dx_stack[i, :, :].ravel()[idxxy]
        dy_stack_masked[i,:] = dy_stack[i, :, :].ravel()[idxxy]
        dx_IQR_masked[i,:] = dx_iqr[i]
        dy_IQR_masked[i,:] = dy_iqr[i]

    # cleanup memory
    del dx_stack, dy_stack

    date_format = "%Y%m%d"
    date0_list = [str(i) for i in np.int32(date0_stack).tolist()]
    dates0 = np.array([dt.datetime.strptime(i, date_format) for i in date0_list])
    date1_list = [str(i) for i in np.int32(date1_stack).tolist()]
    dates1 = np.array([dt.datetime.strptime(i, date_format) for i in date1_list])
    ddates = dates1 - dates0
    ddates_day = np.array([i.days for i in ddates])
    date_list = [str(i) for i in np.int32(date0_stack).tolist()]
    img_dates = np.array([dt.datetime.strptime(i, date_format) for i in date_list])
    img_date_unique = np.unique(img_dates)

    #select only images from same months
    months0 = np.empty(len(dates0))
    months0.fill(np.nan)
    for i in range(len(dates0)):
        months0[i] = dates0[i].month
    dates0_MJJASO_idx, = np.where((months0 >=5) & (months0 <=10))
    dates0_NDJFMA_idx, = np.where((months0 <=4) | (months0 >=11))

    months1 = np.empty(len(dates1))
    months1.fill(np.nan)
    for i in range(len(dates1)):
        months1[i] = dates1[i].month
    dates1_MJJASO_idx, = np.where((months1 >=5) & (months1 <=10))
    dates1_NDJFMA_idx, = np.where((months1 <=4) | (months1 >=11))

    dates_MJJASO_idx = np.intersect1d(dates0_MJJASO_idx, dates1_MJJASO_idx)
    dates_NDJFMA_idx = np.intersect1d(dates0_NDJFMA_idx, dates1_NDJFMA_idx)

    # Run inversion for MJJASO season
    num_date, tbase, date1s, date_list, date12_list, unique_date = prepare_design_matrix_input(date0_stack[dates_MJJASO_idx],
        date1_stack[dates_MJJASO_idx], date_format = "%Y%m%d")
    A, B, refDate = create_design_matrix(len(dates_MJJASO_idx), num_date, tbase, date1s, date_list, refDate=None)
    tbase_diff_MJJASO = np.diff(tbase).reshape(-1, 1)
    print('MJJASO inversion')
    print('\t dx')
    dx_ts_tweights_numba_MJJASO, dx_residuals_tweights_numba_MJJASO, dx_ranks_weights_numba_MJJASO = linalg_tweights_numba(A, dx_stack_masked[dates_MJJASO_idx,:], dx_IQR_masked[dates_MJJASO_idx,:], deltay_stack[dates_MJJASO_idx], deltay_stack_scale, tbase_diff_MJJASO, nre, rcond=1e-5)
    print('\t dy')
    dy_ts_tweights_numba_MJJASO, dy_residuals_tweights_numba_MJJASO, dy_ranks_weights_numba_MJJASO = linalg_tweights_numba(A, dy_stack_masked[dates_MJJASO_idx,:], dy_IQR_masked[dates_MJJASO_idx,:], deltay_stack[dates_MJJASO_idx], deltay_stack_scale, tbase_diff_MJJASO, nre, rcond=1e-5)
    tbase_diff2_MJJASO = np.insert(tbase_diff_MJJASO, 0, 0)

    # Run inversion for NDJFMA season
    num_date, tbase, date1s, date_list, date12_list, unique_date = prepare_design_matrix_input(date0_stack[dates_NDJFMA_idx],
        date1_stack[dates_MJJASO_idx], date_format = "%Y%m%d")
    A, B, refDate = create_design_matrix(len(dates_NDJFMA_idx), num_date, tbase, date1s, date_list, refDate=None)
    tbase_diff_NDJFMA = np.diff(tbase).reshape(-1, 1)
    print('NDJFMA inversion')
    print('\t dx')
    dx_ts_tweights_numba_NDJFMA, dx_residuals_tweights_numba_NDJFMA, dx_ranks_weights_numba_NDJFMA = linalg_tweights_numba(A, dx_stack_masked[dates_NDJFMA_idx,:], dx_IQR_masked[dates_NDJFMA_idx,:], deltay_stack[dates_NDJFMA_idx], deltay_stack_scale, tbase_diff_NDJFMA, nre, rcond=1e-5)
    print('\t dy')
    dy_ts_tweights_numba_NDJFMA, dy_residuals_tweights_numba_NDJFMA, dy_ranks_weights_numba_NDJFMA = linalg_tweights_numba(A, dy_stack_masked[dates_NDJFMA_idx,:], dy_IQR_masked[dates_NDJFMA_idx,:], deltay_stack[dates_NDJFMA_idx], deltay_stack_scale, tbase_diff_NDJFMA, nre, rcond=1e-5)
    tbase_diff2_NDJFMA = np.insert(tbase_diff_NDJFMA, 0, 0)

    # Run inversion for all pairs
    num_date, tbase, date1s, date_list, date12_list, unique_date = prepare_design_matrix_input(date0_stack,
        date1_stack, date_format = "%Y%m%d")
    A, B, refDate = create_design_matrix(len(date0_stack), num_date, tbase, date1s, date_list, refDate=None)
    tbase_diff = np.diff(tbase).reshape(-1, 1)
    print('All dates inversion')
    print('\t dx')
    dx_ts_tweights_numba, dx_residuals_tweights_numba, dx_ranks_weights_numba = linalg_tweights_numba(A, dx_stack_masked, dx_IQR_masked, deltay_stack, deltay_stack_scale, tbase_diff, nre, rcond=1e-5)
    print('\t dy')
    dy_ts_tweights_numba, dy_residuals_tweights_numba, dy_ranks_weights_numba = linalg_tweights_numba(A, dy_stack_masked, dy_IQR_masked, deltay_stack, deltay_stack_scale, tbase_diff, nre, rcond=1e-5)
    tbase_diff2 = np.insert(tbase_diff, 0, 0)

    # dx and dy time series
    fig, ax = plt.subplots(1, 2, figsize=(18, 9), dpi=300)
    # ax[0].plot(np.cumsum(tbase_diff2), np.nanmean(ts_rweights, axis=1), 'k-', label='(1/IQR)')
    ax[0].plot(np.cumsum(tbase_diff2_NDJFMA), np.nanmean(dx_ts_tweights_numba_NDJFMA, axis=1), '-', color='darkred', label='NDJFMA (n=%d dates)'%dx_ts_tweights_numba_NDJFMA.shape[0])
    ax[0].plot(np.cumsum(tbase_diff2_MJJASO), np.nanmean(dx_ts_tweights_numba_MJJASO, axis=1), '-', color='navy', label='MJJASO (n=%d dates)'%dx_ts_tweights_numba_MJJASO.shape[0])
    ax[0].plot(np.cumsum(tbase_diff2), np.nanmean(dx_ts_tweights_numba, axis=1), '-', color='gray', label='all (n=%d dates)'%dx_ts_tweights_numba.shape[0])
    ax[0].set_title('Mean dx offset (n=%d pixels)'%nre, fontsize=14)
    ax[0].set_xlabel('Time [y]')
    ax[0].set_ylabel('Cumulative dx offset [pix]')
    ax[0].legend()
    ax[0].grid()
    ax[1].plot(np.cumsum(tbase_diff2_NDJFMA), np.nanmean(dy_ts_tweights_numba_NDJFMA, axis=1), '-', color='darkred', label='NDJFMA')
    ax[1].plot(np.cumsum(tbase_diff2_MJJASO), np.nanmean(dy_ts_tweights_numba_MJJASO, axis=1), '-', color='navy', label='MJJASO')
    ax[1].plot(np.cumsum(tbase_diff2), np.nanmean(dy_ts_tweights_numba, axis=1), '-', color='gray', label='all')
    ax[1].set_title('Mean dy offset (n=%d pixels)'%nre, fontsize=14)
    ax[1].set_xlabel('Time [y]')
    ax[1].set_ylabel('Cumulative dy offset [pix]')
    ax[1].legend()
    ax[1].grid()
    fig.tight_layout()
    fig.savefig(os.path.join(args.png_out_path, '%s_dx_dy_seasonal_timeseries_scaled_with_different_weights.png'%args.area_name), dpi=300)

    # Export inverted ts to npy files
    if os.path.exists(dx_ts_tweights_numba_fn) is False:
        f = gzip.GzipFile(dx_ts_tweights_numba_fn, "w")
        np.save(file=f, arr=dx_ts_tweights_numba)
        f.close()
        f = None

    if os.path.exists(dy_ts_tweights_numba_fn) is False:
        f = gzip.GzipFile(dy_ts_tweights_numba_fn, "w")
        np.save(file=f, arr=dy_ts_tweights_numba)
        f.close()
        f = None

    if os.path.exists(dx_ts_NDJFMA_tweights_numba_fn) is False:
        f = gzip.GzipFile(dx_ts_NDJFMA_tweights_numba_fn, "w")
        np.save(file=f, arr=dx_ts_tweights_numba_NDJFMA)
        f.close()
        f = None

    if os.path.exists(dy_ts_NDJFMA_tweights_numba_fn) is False:
        f = gzip.GzipFile(dy_ts_NDJFMA_tweights_numba_fn, "w")
        np.save(file=f, arr=dy_ts_tweights_numba_NDJFMA)
        f.close()
        f = None

    if os.path.exists(dx_ts_MJJASO_tweights_numba_fn) is False:
        f = gzip.GzipFile(dx_ts_MJJASO_tweights_numba_fn, "w")
        np.save(file=f, arr=dx_ts_tweights_numba_MJJASO)
        f.close()
        f = None

    if os.path.exists(dy_ts_MJJASO_tweights_numba_fn) is False:
        f = gzip.GzipFile(dy_ts_MJJASO_tweights_numba_fn, "w")
        np.save(file=f, arr=dy_ts_tweights_numba_MJJASO)
        f.close()
        f = None
