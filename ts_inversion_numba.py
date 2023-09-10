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
Run time series inversion on offset pixels via numpy and numba. Takes advantage of multiple cores, but requires memory. Very fast, but only useful for limited number of points (up to 1e5) and limited timesteps (up to 100).
This reads in the offset timeseries and a landslide mask file (e.g., created with generate_landslide_mask.py) and an uncertainty offset file (IQR, generated with create_offset_confidence.py and --method 2).
"""

EXAMPLE = """example:
ts_inversion_numba.py \
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


def linalg_noweights(A, y, tbase_diff, num_pixel, rcond=1e-5):
    #numpy-based least square inversion without weights
    num_date = A.shape[1] + 1
    ts = np.empty((num_date, nre), dtype=np.float32)
    ts.fill(np.nan)
    residuals = np.zeros((A.shape[0], nre), dtype=np.float32)
    ranks = np.empty(nre, dtype=np.float32)
    ranks.fill(np.nan)

    # y = y.reshape(A.shape[0], -1)
    # X, residuals, ranks, _ = np.linalg.lstsq(A, y, rcond=rcond)
    # ts_diff = X * np.tile(tbase_diff, (1, num_pixel))
    # ts[1:, :] = np.cumsum(ts_diff, axis=1)
    print('Run linear inversion on each pixel without weights')
    for i in tqdm.tqdm(range(nre)):
        y2 = y[:,i]
        y2 = np.copy(np.reshape(y2, (A.shape[0], -1)))
        X, residual, ranks[i], _ = np.linalg.lstsq(A, y2,
                                                rcond=rcond)
        if residual.size > 0:
            residuals[:,i] = residual
        else:
            residuals[:,i] = np.squeeze(np.squeeze(A).dot(X))
        ts[0,:] = np.zeros(nre, dtype=np.float32)
        ts_diff = X[:,0] * tbase_diff[:,0]
        ts[1:, i] = np.cumsum(ts_diff)

    return ts, residuals, ranks


def linalg_rweights(A, y, weights, tbase_diff, num_pixel, rcond=1e-5):
    #reverse or reciprocal weights using numpy
    #W = 1. / W
    num_date = A.shape[1] + 1
    ts = np.empty((num_date, nre), dtype=np.float32)
    ts.fill(np.nan)
    residuals = np.empty((A.shape[0], nre), dtype=np.float32)
    residuals.fill(np.nan)
    ranks = np.empty(nre, dtype=np.float32)
    ranks.fill(np.nan)

    #will do pixel-by-pixel inversion, because some pixels may not have data
    # print('Run linear inversion on each pixel with 1 / weights')
    for i in tqdm.tqdm(range(nre)):
        W = weights[:,i]
        W = 1. / W  # use inverse of weight
        W = np.diag(W)
        y2 = y[:,i]
        Aw = np.dot(W, A)
        Bw = np.dot(y2, W)
        X, residual, ranks[i], _ = np.linalg.lstsq(Aw, Bw, rcond=rcond)
        if residual.size > 0:
            residuals[:,i] = residual
        else:
            residuals[:,i] = np.squeeze(A.dot(X))
        ts[0,:] = np.zeros(nre, dtype=np.float32)
        ts_diff = X * tbase_diff[:,0]
        ts[1:, i] = np.cumsum(ts_diff)
    return ts, residuals, ranks

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


@njit(parallel=True)
def linalg_tweights_numba2(A, y, weights, deltay_stack, deltay_stack_scale, residuals, tbase_diff, num_pixel, rcond=1e-5):
    # Test case for using reciprocal IQR scaled with length of time series and scaled with inverse of residual
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
        if np.all(np.isnan(residuals[:,i])):
            W = (1/W) * deltay_stack ** deltay_stack_scale
        else:
            W = (1/W) * deltay_stack ** deltay_stack_scale * (1/residuals[:,i])
        W = np.diag(W).astype(np.float64)
        y2 = y[:,i].astype(np.float64)
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


def linalg_rweights_GPU(A, y, weights, tbase_diff, num_pixel, rcond=1e-5):
    #not faster, unless there are more than 1e5 points
    # makes only sense to run for very large arrays - we don't use this yet
    num_date = A.shape[1] + 1
    ts = cp.empty((num_date, nre), dtype=np.float32)
    ts.fill(np.nan)
    residuals = cp.zeros((A.shape[0], nre), dtype=np.float32)
    ranks = cp.empty(nre, dtype=np.float32)
    ranks.fill(np.nan)
    A_gpu = cp.asarray(A)
    y_gpu = cp.asarray(y)
    weights_gpu = cp.asarray(weights)

    #will do pixel-by-pixel inversion, because some pixels may not have data
    # print('Run linear inversion for each pixel with 1 / IQR (CUDA)')
    for i in tqdm.tqdm(range(nre)):
        weight_sqrt = weights_gpu[:,i]
        weight_sqrt = 1. / weight_sqrt  # use inverse of weight
        y2 = y_gpu[:,i]
        y2 = np.expand_dims(y2, 1)
        weight_sqrt = np.expand_dims(weight_sqrt, 1)
        X, residual, ranks[i], _ = cp.linalg.lstsq(np.multiply(A_gpu, weight_sqrt), np.multiply(y2, weight_sqrt),
                                                rcond=rcond)
        if residual.size > 0:
            residuals[:,i] = residual
        else:
            residuals[:,i] = np.squeeze(np.squeeze(A_gpu).dot(X))
        # ts_diff = np.squeeze(X * np.tile(tbase_diff, (1, num_pixel)))
        ts_diff = cp.asnumpy(X[:,0]) * tbase_diff[:,0]
        ts[1:, i] = np.cumsum(ts_diff)
    res = cp.asnumpy(residuals)
    rank = cp.asnumpy(ranks)
    del X, y2, weight_sqrt, weights_gpu, y_gpu, A_gpu, residuals, ranks
    cp._default_memory_pool.free_all_blocks()
    return ts, res, rank


def linalg_sqrtweights(A, y, weights, tbase_diff, num_pixel, rcond=1e-5):
    #using square root of reciprocal weight
    num_date = A.shape[1] + 1
    ts = np.empty((num_date, nre), dtype=np.float32)
    ts.fill(np.nan)
    residuals = np.empty((A.shape[0], nre), dtype=np.float32)
    residuals.fill(np.nan)
    ranks = np.empty(nre, dtype=np.float32)
    ranks.fill(np.nan)

    #will do pixel-by-pixel inversion, because some pixels may not have data
    # print('Run linear inversion on each pixel with rescaling: sqrt(1/weights)')
    for i in tqdm.tqdm(range(nre)):
        W = weights[:,i]
        W = np.sqrt(1/W)  # use squre root of weight, to faciliate WLS, same as for phase.
        W = np.diag(W)
        y2 = y[:,i]
        Aw = np.dot(W, A)
        Bw = np.dot(y2, W)
        X, residual, ranks[i], _ = np.linalg.lstsq(Aw, Bw, rcond=rcond)

        if residual.size > 0:
            residuals[:,i] = residual
        else:
            residuals[:,i] = np.squeeze(np.squeeze(A).dot(X))
        ts[0,:] = np.zeros(nre, dtype=np.float32)
        ts_diff = X * tbase_diff[:,0]
        ts[1:, i] = np.cumsum(ts_diff)
    return ts, residuals, ranks


def linalg_weights_norescaling(A, y, weights, tbase_diff, num_pixel, rcond=1e-5):
    #uses weights (IQR) without scaling
    num_date = A.shape[1] + 1
    ts = np.empty((num_date, nre), dtype=np.float32)
    ts.fill(np.nan)
    residuals = np.empty((A.shape[0], nre), dtype=np.float32)
    residuals.fill(np.nan)
    ranks = np.empty(nre, dtype=np.float32)
    ranks.fill(np.nan)

    #will do pixel-by-pixel inversion, because some pixels may not have data
    # print('Run linear inversion on each pixel with weights (no rescaling)')
    for i in tqdm.tqdm(range(nre)):
        W = weights[:,i]
        W = np.diag(W)
        y2 = y[:,i]
        Aw = np.dot(W, A)
        Bw = np.dot(y2, W)
        X, residual, ranks[i], _ = np.linalg.lstsq(Aw, Bw, rcond=rcond)

        if residual.size > 0:
            residuals[:,i] = residual
        else:
            residuals[:,i] = np.squeeze(np.squeeze(A).dot(X))

        ts[0,:] = np.zeros(nre, dtype=np.float32)
        ts_diff = X * tbase_diff[:,0]
        ts[1:, i] = np.cumsum(ts_diff)
    return ts, residuals, ranks


def linalg_tweights(A, y, weights, deltay_stack, deltay_stack_scale, tbase_diff, num_pixel, rcond=1e-5):
    # reciprocal weights scaled by time step/duration
    num_date = A.shape[1] + 1
    ts = np.empty((num_date, nre), dtype=np.float32)
    ts.fill(np.nan)
    residuals = np.empty((A.shape[0], nre), dtype=np.float32)
    residuals.fill(np.nan)
    ranks = np.empty(nre, dtype=np.float32)
    ranks.fill(np.nan)

    #will do pixel-by-pixel inversion, because some pixels may not have data
    # print('Run linear inversion on each pixel scaled with time difference 1/(weight_sqrt)*deltay_stack[i]**deltay_stack_scale')
    for i in tqdm.tqdm(range(nre)):
        W = weights[:,i]
        # weight_sqrt[np.isnan(weight_sqrt)] = 100.
        # weight_sqrt[weight_sqrt < 0.005] = 0.005
        W = (1/W) * deltay_stack**deltay_stack_scale
        W = np.diag(W)
        y2 = y[:,i]
        Aw = np.dot(W, A)
        Bw = np.dot(y2, W)
        X, residual, ranks[i], _ = np.linalg.lstsq(Aw, Bw, rcond=rcond)

        if residual.size > 0:
            residuals[:,i] = residual
        else:
            residuals[:,i] = np.squeeze(np.squeeze(A).dot(X))
        ts[0,:] = np.zeros(nre, dtype=np.float32)
        ts_diff = X * tbase_diff[:,0]
        ts[1:, i] = np.cumsum(ts_diff)
    return ts, residuals, ranks


def linalg_tweights2(A, y, weights, deltay_stack, deltay_stack_scale, residuals, tbase_diff, num_pixel, rcond=1e-5):
    # Test case with reciprocal weights scaled by timestep/duration and inverse of residuals
    num_date = A.shape[1] + 1
    ts = np.empty((num_date, nre), dtype=np.float32)
    ts.fill(np.nan)
    residuals = np.empty((A.shape[0], nre), dtype=np.float32)
    residuals.fill(np.nan)
    ranks = np.empty(nre, dtype=np.float32)
    ranks.fill(np.nan)

    #will do pixel-by-pixel inversion, because some pixels may not have data
    # print('Run linear inversion on each pixel scaled with time difference 1/(weight_sqrt)*deltay_stack[i]**deltay_stack_scale')
    for i in tqdm.tqdm(range(nre)):
        W = weights[:,i]
        if np.all(np.isnan(residuals[:,i])):
            W = (1/W) * deltay_stack **deltay_stack_scale
        else:
            W = (1/W) * deltay_stack **deltay_stack_scale * (1/residuals[:,i])
        W = np.diag(W)
        y2 = y[:,i]
        Aw = np.dot(W, A)
        Bw = np.dot(y2, W)
        X, residual, ranks[i], _ = np.linalg.lstsq(Aw, Bw, rcond=rcond)

        if residual.size > 0:
            residuals[:,i] = residual
        else:
            residuals[:,i] = np.squeeze(np.squeeze(A).dot(X))
        ts[0,:] = np.zeros(nre, dtype=np.float32)
        ts_diff = X * tbase_diff[:,0]
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
    dx_ts_rweights_numba_fn = area_name + '%s_dx_ts_rweights.npy.gz'%args.area_name
    dy_ts_rweights_numba_fn = area_name + '%s_dy_ts_rweights.npy.gz'%args.area_name
    dx_ts_rweights2_numba_fn = area_name + '%s_dx_ts_rweights2.npy.gz'%args.area_name
    dy_ts_rweights2_numba_fn = area_name + '%s_dy_ts_rweights2.npy.gz'%args.area_name
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

    # DEBUG
    # # plot median confidence values - only for image data
    # conf_median = cc.nanmedian_numba(conf)
    # conf_std = cc.nanstd_numba(conf)
    # fig, ax = plt.subplots(1, 2, figsize=(18, 9), dpi=300)
    # im0 = ax[0].imshow(conf_median, vmin=np.nanpercentile(conf_median,2), vmax=np.nanpercentile(conf_median,98), cmap='viridis')
    # ax[0].set_title('Median confidence values (n=%d timesteps)'%conf.shape[0], fontsize=14)
    # cb0 = fig.colorbar(im0, ax=ax[0], location='bottom', pad=0.1)
    # cb0.set_label('Median confidence [px]')
    #
    # im1 = ax[1].imshow(conf_std, vmin=np.nanpercentile(conf_std,2), vmax=np.nanpercentile(conf_std,98), cmap='magma')
    # ax[1].set_title('Std. Dev. confidence values (n=%d timesteps)'%conf.shape[0], fontsize=14)
    # cb1 = fig.colorbar(im1, ax=ax[1], location='bottom', pad=0.1)
    # cb1.set_label('std. dev. confidence [px]')
    # fig.tight_layout()
    # fig.savefig('ConfidenceValues_median_stddev.png', dpi=300)

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
    # date_list = [str(i) for i in np.int32(date1s).tolist()]
    # img_date_unique = np.array([dt.datetime.strptime(i, date_format) for i in date_list])

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


    # prepare and create design_matrix
    num_date, tbase, date1s, date_list, date12_list, unique_date = prepare_design_matrix_input(date0_stack, date1_stack, date_format = "%Y%m%d")
    A, B, refDate = create_design_matrix(num_ifgram, num_date, tbase, date1s, date_list, refDate=None)
    tbase_diff = np.diff(tbase).reshape(-1, 1)
    # print('refDate: ',refDate)
    # print('Shape Design Matrix: ', A_nrn1.shape)
    # print()
    # print('Date list (without reference/first date:')
    # print(date_list[1::])
    # print()
    # print('Design matrix for displacement data: ')
    # plt.imshow(A), plt.colorbar(), plt.title('Design Matrix A'), plt.show()

    # Plot different weights - only for dx
    fig, ax = plt.subplots(1, 2, figsize=(18, 9), dpi=300)
    ax[0].plot(dx_iqr, 1/dx_iqr, 'x', ms=3, color='k', label='1/dx_iqr')
    ax[0].plot(dx_iqr, np.sqrt(1/dx_iqr), 'o', ms=2, color='navy', label='np.sqrt(1/dx_iqr)')
    ax[0].plot(dx_iqr, (1/dx_iqr)*deltay_stack**deltay_stack_scale, 's', ms=2,color='darkred', label='(1/dx_iqr)*deltay_stack**deltay_stack_scale')
    # ax[0].plot(dx_iqr, 1/(dx_iqr**2), 'o', ms=2, color='darkred', label='1/(dx_iqr**2)')
    ax[0].set_title('dx IQR scaling (n=%d timesteps)'%dx_iqr.shape[0], fontsize=14)
    ax[0].set_xlabel('dx IQR')
    ax[0].set_ylabel('weight (rescaled IQR)')
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(deltay_stack, dx_iqr, 'o', ms=2, color='darkred', label='dx_iqr')
    ax12 = ax[1].twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax12.set_ylabel('weight (rescaled IQR)', color=color)  # we already handled the x-label with ax1
    ax12.plot(deltay_stack, (1/dx_iqr)*deltay_stack**deltay_stack_scale, 'o', ms=2, color=color, label='(1/dx_iqr)*deltay_stack**deltay_stack_scale')
    ax12.tick_params(axis='y', labelcolor=color)
    ax[1].set_title('dx IQR vs. $\Delta$ time (n=%d timesteps)'%dx_iqr.shape[0], fontsize=14)
    ax[1].set_ylabel('dx IQR')
    ax[1].set_xlabel('$\Delta$ time [y]')
    ax[1].grid()
    ax[1].legend()
    ax12.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.png_out_path, '%s_dx_IQR_scaling.png'%args.area_name), dpi=300)

    tbase_diff2 = np.insert(tbase_diff, 0, 0)

    # least square regression with weights = 1/confidence (useful for confidence derived from IQR or std. dev.)
    # ts_tweights, residuals_tweights, ranks_tweights = linalg_rweights(A, dx_stack_masked, dx_IQR_masked, tbase_diff, nre, rcond=None)
    print('Run linear inversion on each pixel with 1 / weights scaled by measurement duration (1/IQR)*deltay_stack**deltay_stack_scale')
    # weight_sqrt = 1/(dx_iqr)*deltay_stack**deltay_stack_scale
    # ts_tweights, residuals_tweights, ranks_tweights = linalg_tweights(A, dx_stack_masked, dx_IQR_masked, deltay_stack, deltay_stack_scale, tbase_diff, nre, rcond=1e-5)
    print('\t dx')
    dx_ts_tweights_numba, dx_residuals_tweights_numba, dx_ranks_weights_numba = linalg_tweights_numba(A, dx_stack_masked, dx_IQR_masked, deltay_stack, deltay_stack_scale, tbase_diff, nre, rcond=1e-5)
    print('\t dy')
    dy_ts_tweights_numba, dy_residuals_tweights_numba, dy_ranks_weights_numba = linalg_tweights_numba(A, dy_stack_masked, dy_IQR_masked, deltay_stack, deltay_stack_scale, tbase_diff, nre, rcond=1e-5)

    # Plot ts_rweights and residuals weights
    # fig, ax = plt.subplots(1, 2, figsize=(18, 9), dpi=300)
    # diff=np.empty(ts_rweights.shape[1])
    # diff.fill(np.nan)
    # for i in range(ts_rweights.shape[1]):
    #     diff[i] = np.nansum( (ts_rweights[:,i] - ts_rweights_numba[:,i])**2 )
    # i, = np.where(np.nanmin(diff)==diff)
    # i = i[0]
    # ax[0].plot(np.cumsum(tbase_diff2), np.squeeze(ts_rweights[:,i]), 'x', ms=5, color='k', label='MIN rweights i=%d'%i)
    # ax[0].plot(np.cumsum(tbase_diff2), np.squeeze(ts_rweights_numba[:,i]), 'o', ms=3, color='k', label='MIN rweights numba i=%d'%i)
    # i, = np.where(np.nanmax(diff)==diff)
    # ax[0].plot(np.cumsum(tbase_diff2), np.squeeze(ts_rweights[:,i]), 'x', ms=5, color='darkred', label='MAX rweights i=%d'%i)
    # ax[0].plot(np.cumsum(tbase_diff2), np.squeeze(ts_rweights_numba[:,i]), 'o', ms=3, color='darkred', label='MAX rweights numba i=%d'%i)
    # ax[0].set_title('TS with rweights (n=%d timesteps)'%dx_iqr.shape[0], fontsize=14)
    # ax[0].set_xlabel('time [y]')
    # ax[0].set_ylabel('ts rweights')
    # ax[0].grid()
    # ax[0].legend()

    # Plot residuals and time series
    # residual_range = np.nanmax(dx_residuals_tweights, axis=1) - np.nanmin(dx_residuals_tweights, axis=1)
    dx_residual_range_numba = np.nanmax(dx_residuals_tweights_numba, axis=1) - np.nanmin(dx_residuals_tweights_numba, axis=1)
    dy_residual_range_numba = np.nanmax(dy_residuals_tweights_numba, axis=1) - np.nanmin(dy_residuals_tweights_numba, axis=1)
    fig, ax = plt.subplots(1, 3, figsize=(18, 9), dpi=300)
    # ax[0].errorbar(dates0[0:15], np.nanmean(residuals_rweights_numba, axis=1)[0:15], np.nanstd(residuals_rweights_numba, axis=1)[0:15], color='navy', label='residuals numba')
    # ax[0].errorbar(dates0[0:15], np.nanmean(residuals_rweights, axis=1)[0:15], np.nanstd(residuals_rweights, axis=1)[0:15], ms=5, color='k', label='residuals')
    # ax[0].plot(dates0, np.nanmean(residuals_rweights_numba, axis=1), 'o', ms=3, color='navy', label='residuals numba')
    # ax[0].plot(dates0, np.nanmean(dx_residuals_tweights, axis=1), 'x', ms=3, color='k', label='residuals')
    im0 = ax[0].scatter(dates0, dx_residual_range_numba, c=deltay_stack, s=3, label='residual range')
    cb0 = fig.colorbar(im0, ax=ax[0], location='bottom', pad=0.1)
    cb0.set_label('Time difference [y]')
    ax[0].set_title('dx residuals for each offset calculation (n=%d timesteps)'%dx_iqr.shape[0], fontsize=14)
    ax[0].set_xlabel('Starting Date')
    ax[0].set_ylabel('mean dx residuals of all pixels per time step')
    ax[0].grid()
    ax[0].legend()
    ax[0].set_ylim([0, 6])
    ax[1].plot(deltay_stack, np.nanmean(dx_residuals_tweights_numba, axis=1), 'o', ms=3, color='navy', label='dx residuals numba')
    ax[1].plot(deltay_stack, np.nanmean(dy_residuals_tweights_numba, axis=1), 'x', ms=3, color='darkred', label='dy residuals numba')
    # ax[1].plot(deltay_stack, np.nanmean(dx_residuals_tweights, axis=1), 'x', ms=5, color='k', label='residuals')
    # ax[1].plot(deltay_stack, np.nanmean(dx_residuals_tweights2, axis=1), 'o', ms=5, color='darkred', label='residuals2')
    # ax[1].plot(deltay_stack, residual_range, 's', ms=3, color='darkred', label='residual range')
    ax[1].set_title('Timestep and residuals (n=%d timesteps)'%deltay_stack.shape[0], fontsize=14)
    ax[1].set_ylabel('residuals')
    ax[1].set_xlabel('$\Delta$ time [y]')
    ax[1].grid()
    ax[1].legend()
    ax[2].plot(dx_iqr, dx_residual_range_numba, 'o', ms=3, color='navy', label='residuals numba')
    ax[2].plot(dy_iqr, dy_residual_range_numba, 'x', ms=3, color='darkred', label='residuals numba')
    # ax[2].plot(dx_iqr, residual_range, 'x', ms=3, color='k', label='residuals')
    ax[2].set_title('IQR and residuals (n=%d timesteps)'%dx_iqr.shape[0], fontsize=14)
    ax[2].set_ylabel('residual range ')
    ax[2].set_xlabel('IQR')
    ax[2].grid()
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.png_out_path,'%s_ts_residuals.png'%args.area_name), dpi=300)

    # #rerun inversion with added weights from residuals
    # not useful, removing this step
    # print('Re-run inversion with IQR and residual scaling')
    # ts_tweights2, residuals_tweights2, ranks_tweights2 = linalg_tweights2(A, dx_stack_masked, dx_IQR_masked, deltay_stack, deltay_stack_scale, residuals_tweights, tbase_diff, nre, rcond=1e-5)
    # ts_tweights2_numba, residuals_tweights2_numba, ranks_tweights2_numba = linalg_tweights_numba2(A, dx_stack_masked, dx_IQR_masked, deltay_stack, deltay_stack_scale, residuals_tweights_numba, tbase_diff, nre, rcond=1e-5)
    #
    # # Plot residuals from both runs and time series
    # residual_range2 = np.nanmax(residuals_tweights2, axis=1) - np.nanmin(residuals_tweights2, axis=1)
    # residual_range2_numba = np.nanmax(residuals_tweights2_numba, axis=1) - np.nanmin(residuals_tweights2_numba, axis=1)
    # fig, ax = plt.subplots(1, 3, figsize=(18, 9), dpi=300)
    # date0_list = [str(i) for i in np.int32(date0_stack).tolist()]
    # dates0 = np.array([dt.datetime.strptime(i, date_format) for i in date0_list])
    # ax[0].plot(dates0, residual_range, 'x', ms=3, color='k', label='residual range')
    # ax[0].plot(dates0, residual_range2, 'o', ms=3, color='navy', label='residual range 2')
    # ax[0].set_title('TS with weights (n=%d timesteps)'%dx_iqr.shape[0], fontsize=14)
    # ax[0].set_xlabel('Starting Date')
    # ax[0].set_ylabel('mean residuals of all pixels per time step')
    # ax[0].set_ylim([0, 6])
    # ax[0].grid()
    # ax[0].legend()
    # # ax[1].plot(deltay_stack, np.nanmean(residuals_tweights2_numba, axis=1), 'o', ms=5, color='navy', label='residuals numba')
    # ax[1].plot(deltay_stack, np.nanmean(residuals_tweights, axis=1), 'x', ms=5, color='k', label='residuals')
    # ax[1].plot(deltay_stack, np.nanmean(residuals_tweights2, axis=1), 'o', ms=5, color='darkred', label='residuals 2')
    # ax[1].plot(deltay_stack, residual_range, 's', ms=5, color='darkred', label='residual range')
    # ax[1].set_title('Timestep and residuals (n=%d timesteps)'%deltay_stack.shape[0], fontsize=14)
    # ax[1].set_ylabel('residuals ')
    # ax[1].set_xlabel('$\Delta$ time [y]')
    # ax[1].grid()
    # ax[1].legend()
    # ax[2].plot(dx_iqr, residual_range2_numba, 'o', ms=3, color='navy', label='residuals')
    # ax[2].plot(dx_iqr, residual_range2, 'x', ms=5, color='k', label='residuals numba')
    # ax[2].set_title('IQR and residuals (n=%d timesteps)'%dx_iqr.shape[0], fontsize=14)
    # ax[2].set_ylabel('residual range ')
    # ax[2].set_xlabel('IQR')
    # ax[2].grid()
    # ax[2].legend()
    # fig.tight_layout()
    # fig.savefig(os.path.join(args.png_out_path, 'ts_residuals2.png'), dpi=300)

    # # np.sqrt(1/weight_sqrt)
    # print('Run linear inversion on each pixel with sqrt(1 / weights)')
    # ts_sqrtweights, residuals_sqrtweights, ranks_sqrtweights = linalg_sqrtweights(A, dx_stack_masked, dx_IQR_masked, tbase_diff, nre, rcond=1e-5)

    # print('Run linear inversion on each pixel with weight = IQR')
    # # weights = dx_IQR
    # ts_norescalingweights, residuals_norescalingweights, ranks_norescalingweights = linalg_weights_norescaling(A, dx_stack_masked, dx_IQR_masked, tbase_diff, nre, rcond=1e-5)

    print('Run linear inversion on each pixel with 1 / IQR and 1/IQR**2')
    # ts_rweights, residuals_rweights, ranks_rweights = linalg_rweights(A, dx_stack_masked, dx_IQR_masked, tbase_diff, nre, rcond=1e-5)
    print('\t dx')
    dx_ts_rweights_numba, dx_residuals_rweights_numba, dx_ranks_rweights_numba = linalg_rweights_numba(A, dx_stack_masked, dx_IQR_masked, tbase_diff, nre, rcond=1e-5)
    dx_ts_rweights2_numba, dx_residuals_rweights2_numba, dx_ranks_rweights2_numba = linalg_rweights2_numba(A, dx_stack_masked, dx_IQR_masked, tbase_diff, nre, rcond=1e-5)
    print('\t dy')
    dy_ts_rweights_numba, dy_residuals_rweights_numba, dy_ranks_rweights_numba = linalg_rweights_numba(A, dy_stack_masked, dy_IQR_masked, tbase_diff, nre, rcond=1e-5)
    dy_ts_rweights2_numba, dy_residuals_rweights2_numba, dy_ranks_rweights2_numba = linalg_rweights2_numba(A, dy_stack_masked, dy_IQR_masked, tbase_diff, nre, rcond=1e-5)

    # no weights
    print('Run linear inversion on each pixel with no weights')
    print('\t dx')
    dx_ts_noweights_numba, dx_residuals_noweights_numba, dx_ranks_noweights_numba = linalg_noweights_numba(A, dx_stack_masked, tbase_diff, nre, rcond=1e-5)
    print('\t dy')
    dy_ts_noweights_numba, dy_residuals_noweights_numba, dy_ranks_noweights_numba = linalg_noweights_numba(A, dy_stack_masked, tbase_diff, nre, rcond=1e-5)

    tbase_diff2 = np.insert(tbase_diff, 0, 0)

    # residual vs. time span
    fig, ax = plt.subplots(1, 2, figsize=(18, 9), dpi=300)
    ax[0].plot(ddates_day, np.nanmean(dx_residuals_rweights_numba, axis=1), 'k+', label='numba (1/IQR)')
    ax[0].plot(ddates_day, np.nanmean(dx_residuals_tweights_numba, axis=1), 'o', ms=3, color='darkred', label='numba (1/IQR)*deltay_stack**deltay_stack_scale')
    ax[0].plot(ddates_day, np.nanmean(dx_residuals_rweights2_numba, axis=1), 's', ms=3, color='navy', label='numba (1/IQR**2)')
    ax[0].set_title('Mean Residual from each pixel for each image pair (n=%d)'%dx_residuals_rweights_numba.shape[0], fontsize=14)
    ax[0].set_xlabel('Time Difference [days]')
    ax[0].set_ylabel('dx residual')
    ax[0].legend()
    ax[0].grid()

    #first image date vs. residual
    ax[1].plot(img_dates, np.nanmean(dx_residuals_rweights_numba, axis=1), 'k+', label='numba (1/IQR)')
    ax[1].plot(img_dates, np.nanmean(dx_residuals_tweights_numba, axis=1), 'o', ms=3, color='darkred', label='numba (1/IQR)*deltay_stack**deltay_stack_scale')
    ax[1].plot(img_dates, np.nanmean(dx_residuals_rweights2_numba, axis=1), 's', ms=3, color='navy', label='numba (1/IQR**2)')
    # ax12 = ax[1].twinx()
    # ax12.plot(img_date_unique, np.nanmean(ts_norescalingweights, axis=1), '-', color='navy', label='no weights')
    ax[1].set_title('Mean Residual for each image pair (n=%d)'%dx_residuals_rweights_numba.shape[0], fontsize=14)
    ax[1].set_xlabel('First image date')
    ax[1].set_ylabel('dx residual')
    ax[1].legend()
    ax[1].grid()
    fig.tight_layout()
    fig.savefig(os.path.join(args.png_out_path, '%s_ts_residuals_with_different_weights.png'%args.area_name), dpi=300)

    # # Generating Debugging and test plots
    # fig, ax = plt.subplots(1, 3, figsize=(18, 9), dpi=300)
    # ax[0].plot(np.cumsum(tbase_diff2), np.nanmean(ts_rweights, axis=1), 'k-', label='(1/IQR)')
    # ax[0].plot(np.cumsum(tbase_diff2), np.nanmean(ts_rweights_numba, axis=1), '-', color='gray', label='numba (1/IQR)')
    # ax[0].plot(np.cumsum(tbase_diff2), np.nanmean(ts_rweights2_numba, axis=1), '-', color='green', label='numba (1/IQR**2)')
    # # ax[0].plot(np.cumsum(tbase_diff2), np.nanmean(ts_sqrtweights, axis=1), '-', color='darkred', label='sqrt(1/IQR)')
    # # ax[0].plot(np.cumsum(tbase_diff2), np.nanmean(ts_norescalingweights, axis=1), '-', color='magenta', label='IQR (unscaled)')
    # ax[0].plot(np.cumsum(tbase_diff2), np.nanmean(ts_tweights, axis=1), '-', color='darkred', label='(1/IQR)*deltay_stack**deltay_stack_scale')
    # # ax[0].plot(np.cumsum(tbase_diff2), np.nanmean(ts_tweights2, axis=1), '-', color='darkred', label='residuals * (1/IQR)*deltay_stack**deltay_stack_scale')
    # ax[0].plot(np.cumsum(tbase_diff2), np.nanmean(dx_ts_tweights_numba, axis=1), '-', color='navy', label='numba (1/IQR)*deltay_stack**deltay_stack_scale')
    # ax[0].set_title('Mean offset (n=%d)'%nre, fontsize=14)
    # ax[0].set_xlabel('Time [y]')
    # ax[0].set_ylabel('Cumulative offset [pix]')
    # ax[0].legend()
    # ax[0].grid()
    # for i in np.arange(0, ts_rweights.shape[1], 2000):
    #     if i == 0:
    #         ax[1].plot(np.cumsum(tbase_diff2), ts_tweights[:,i], '-', color='darkred', lw=0.5, label='(1/IQR)*deltay_stack**deltay_stack_scale')
    #         ax[1].plot(np.cumsum(tbase_diff2), dx_ts_tweights_numba[:,i], '-', color='navy', lw=0.5, label='numba')
    #         # ax[1].plot(np.cumsum(tbase_diff2), ts_noweights[:,i], '-', color='navy', lw=0.5, label='no weights')
    #     else:
    #         ax[1].plot(np.cumsum(tbase_diff2), ts_tweights[:,i], '-', color='darkred', lw=0.5)
    #         ax[1].plot(np.cumsum(tbase_diff2), dx_ts_tweights_numba[:,i], '-', color='navy', lw=0.5)
    #         # ax[1].plot(np.cumsum(tbase_diff2), ts_noweights[:,i], '-', color='navy', lw=0.5)
    # ax[1].set_xlabel('Time [y]')
    # ax[1].set_ylabel('Cumulative offset [pix]')
    # ax[1].set_title('Individual offsets (n=%d)'%nre, fontsize=14)
    # ax[1].legend()
    # # for i in np.arange(0, ts.shape[1], 1000):
    # #     if i == 0:
    # #         ax[2].plot(np.cumsum(tbase_diff), ts_rweights[1:,i] - ts_noweights[1:,i], '-', color='navy', lw=1, label='weights - noweights')
    # #         ax[2].plot(np.cumsum(tbase_diff), ts_rweights[1:,i] - ts_sqrtweights[1:,i], '-', color='darkred', lw=1, label='weights1 - weights2')
    # #     else:
    # #         ax[2].plot(np.cumsum(tbase_diff), ts_rweights[1:,i] - ts_noweights[1:,i], '-', color='navy', lw=1)
    # #         ax[2].plot(np.cumsum(tbase_diff), ts_rweights[1:,i] - ts_sqrtweights[1:,i], '-', color='darkred', lw=1)
    # ax[2].plot(np.cumsum(tbase_diff2), np.nanmean(ts_rweights, axis=1) - np.nanmean(ts_sqrtweights, axis=1),
    #     '-', color='darkred', label='1 / IQR - sqrt(1/IQR)')
    # # ax[2].plot(np.cumsum(tbase_diff2), np.nanmean(ts_rweights, axis=1) - np.nanmean(ts_norescalingweights, axis=1),
    # #     '-', color='magenta', label='1 / IQR - IQR')
    # ax[2].plot(np.cumsum(tbase_diff2), np.nanmean(ts_rweights, axis=1) - np.nanmean(ts_tweights, axis=1),
    #     '-', color='green', label='1 / IQR - 1/(IQR)*deltay_stack**deltay_stack_scale')
    # ax[2].plot(np.cumsum(tbase_diff2), np.nanmean(ts_rweights, axis=1) - np.nanmean(ts_noweights, axis=1),
    #     '-', color='navy', label='1 / IQR - no weights')
    # ax[2].set_xlabel('Time [y]')
    # ax[2].set_ylabel('$\Delta$ cumulative offset weights - noweights [pix]')
    # ax[2].set_title('Mean cumulative offsets differences (n=%d)'%nre, fontsize=14)
    # ax[2].legend()
    # ax[2].grid()
    # fig.tight_layout()
    # fig.savefig(os.path.join(args.png_out_path, 'timeseries_scaled_with_different_weights.png'), dpi=300)

    # dx and dy time series
    fig, ax = plt.subplots(1, 2, figsize=(18, 9), dpi=300)
    # ax[0].plot(np.cumsum(tbase_diff2), np.nanmean(ts_rweights, axis=1), 'k-', label='(1/IQR)')
    ax[0].plot(np.cumsum(tbase_diff2), np.nanmean(dx_ts_rweights_numba, axis=1), '-', color='gray', label='numba (1/IQR)')
    ax[0].plot(np.cumsum(tbase_diff2), np.nanmean(dx_ts_rweights2_numba, axis=1), '-', color='green', label='numba (1/IQR**2)')
    # ax[0].plot(np.cumsum(tbase_diff2), np.nanmean(ts_sqrtweights, axis=1), '-', color='darkred', label='sqrt(1/IQR)')
    # ax[0].plot(np.cumsum(tbase_diff2), np.nanmean(ts_norescalingweights, axis=1), '-', color='magenta', label='IQR (unscaled)')
    # ax[0].plot(np.cumsum(tbase_diff2), np.nanmean(ts_tweights, axis=1), '-', color='lightblue', label='(1/IQR)*deltay_stack**deltay_stack_scale')
    # ax[0].plot(np.cumsum(tbase_diff2), np.nanmean(ts_tweights2, axis=1), '-', color='darkred', label='residuals * (1/IQR)*deltay_stack**deltay_stack_scale')
    ax[0].plot(np.cumsum(tbase_diff2), np.nanmean(dx_ts_tweights_numba, axis=1), '-', color='darkblue', label='numba (1/IQR)*deltay_stack**deltay_stack_scale')
    ax[0].set_title('Mean dx offset (n=%d)'%nre, fontsize=14)
    ax[0].set_xlabel('Time [y]')
    ax[0].set_ylabel('Cumulative dx offset [pix]')
    ax[0].legend()
    ax[0].grid()
    ax[1].plot(np.cumsum(tbase_diff2), np.nanmean(dy_ts_rweights_numba, axis=1), '-', color='gray', label='numba (1/IQR)')
    ax[1].plot(np.cumsum(tbase_diff2), np.nanmean(dy_ts_rweights2_numba, axis=1), '-', color='green', label='numba (1/IQR**2)')
    ax[1].plot(np.cumsum(tbase_diff2), np.nanmean(dy_ts_tweights_numba, axis=1), '-', color='darkblue', label='numba (1/IQR)*deltay_stack**deltay_stack_scale')
    ax[1].set_title('Mean dy offset (n=%d)'%nre, fontsize=14)
    ax[1].set_xlabel('Time [y]')
    ax[1].set_ylabel('Cumulative dy offset [pix]')
    ax[1].legend()
    ax[1].grid()
    fig.tight_layout()
    fig.savefig(os.path.join(args.png_out_path, '%s_dx_dy_timeseries_scaled_with_different_weights.png'%args.area_name), dpi=300)

    # Export inverted ts to npy files
    if os.path.exists(dx_ts_rweights_numba_fn) is False:
        f = gzip.GzipFile(dx_ts_rweights_numba_fn, "w")
        np.save(file=f, arr=dx_ts_rweights_numba)
        f.close()
        f = None

    if os.path.exists(dy_ts_rweights_numba_fn) is False:
        f = gzip.GzipFile(dy_ts_rweights_numba_fn, "w")
        np.save(file=f, arr=dy_ts_rweights_numba)
        f.close()
        f = None
