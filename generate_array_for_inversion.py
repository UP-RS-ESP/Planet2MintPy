#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Bodo Bookhagen and Ariane Mueting

"""
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
    num_date = A.shape[1] + 1
    ts = np.zeros((num_date, nre), dtype=np.float32)
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
        # ts_diff = np.squeeze(X * np.tile(tbase_diff, (1, num_pixel)))
        ts_diff = X[:,0] * tbase_diff[:,0]
        ts[1:, i] = np.cumsum(ts_diff)

    return ts, residuals, ranks


# @njit(parallel=True)
def linalg_weights(A, y, weights, tbase_diff, num_pixel, rcond=1e-5):
    num_date = A.shape[1] + 1
    ts = np.zeros((num_date, nre), dtype=np.float32)
    residuals = np.zeros((A.shape[0], nre), dtype=np.float32)
    ranks = np.empty(nre, dtype=np.float32)
    ranks.fill(np.nan)

    #will do pixel-by-pixel inversion, because some pixels may not have data
    print('Run linear inversion on each pixel with 1 / weights')
    for i in tqdm.tqdm(range(nre)):
        weight_sqrt = weights[:,i]
        weight_sqrt[np.isnan(weight_sqrt)] = 100.
        weight_sqrt[weight_sqrt < 0.005] = 0.005
        weight_sqrt = 1. / weight_sqrt  # use inverse of weight
        y2 = y[:,i]
        # y = np.copy(np.reshape(y2, (A.shape[0], -1)))
        y2 = np.expand_dims(y2, 1)
        # weight_sqrt = np.copy(weight_sqrt.reshape(A.shape[0], -1))
        weight_sqrt = np.expand_dims(weight_sqrt, 1)
        X, residual, ranks[i], _ = np.linalg.lstsq(np.multiply(A, weight_sqrt), np.multiply(y2, weight_sqrt),
                                                rcond=rcond)
        if residual.size > 0:
            residuals[:,i] = residual
        else:
            residuals[:,i] = np.squeeze(np.squeeze(A).dot(X))
        # ts_diff = np.squeeze(X * np.tile(tbase_diff, (1, num_pixel)))
        ts_diff = X[:,0] * tbase_diff[:,0]
        ts[1:, i] = np.cumsum(ts_diff)
    return ts, residuals, ranks


# @njit(parallel=True)
def linalg_weights_bs(A, y, weights, idx_stack0, idx_stack1, tbase_diff, num_pixel, rcond=1e-5):
    num_date = A.shape[1] + 1
    ts = np.zeros((num_date, nre), dtype=np.float32)
    residuals = np.zeros((A.shape[0], nre), dtype=np.float32)
    ranks = np.empty(nre, dtype=np.float32)
    ranks.fill(np.nan)

    #will do pixel-by-pixel inversion, because some pixels may not have data
    print('Run linear inversion on each pixel with 1 / weights')
    for i in tqdm.tqdm(range(nre)):
        weight_sqrt = weights[:,i]
        weight_sqrt[np.isnan(weight_sqrt)] = 100.
        weight_sqrt[weight_sqrt < 0.005] = 0.005
        weight_sqrt = 1. / weight_sqrt  # use squre root of weight, to faciliate WLS, same as for phase.
        y2 = y[:,i]

        #remove boot-strapped date from weight and y matrix
        weight_sqrt =  np.delete(weight_sqrt, idx_stack0)
        weight_sqrt =  np.delete(weight_sqrt, idx_stack1)
        weight_sqrt = np.copy(weight_sqrt.reshape(A.shape[0], -1))
        y2 = np.delete(y2, idx_stack0)
        y2 = np.delete(y2, idx_stack1)
        y2 = y2.reshape(A.shape[0], -1)

        X, residual, ranks[i], _ = np.linalg.lstsq(np.multiply(A, weight_sqrt), np.multiply(y2, weight_sqrt),
                                                rcond=rcond)

        if residual.size > 0:
            residuals[:,i] = residual
        else:
            residuals[:,i] = np.squeeze(np.squeeze(A).dot(X))
        # ts_diff = np.squeeze(X * np.tile(tbase_diff, (1, num_pixel)))
        ts_diff = X[:,0] * tbase_diff[:,0]
        ts[1:, i] = np.cumsum(ts_diff)
    return ts, residuals, ranks

def linalg_weights2(A, y, weights, tbase_diff, num_pixel, rcond=1e-5):
    num_date = A.shape[1] + 1
    ts = np.zeros((num_date, nre), dtype=np.float32)
    residuals = np.zeros((A.shape[0], nre), dtype=np.float32)
    ranks = np.empty(nre, dtype=np.float32)
    ranks.fill(np.nan)

    #will do pixel-by-pixel inversion, because some pixels may not have data
    print('Run linear inversion on each pixel with sqrt(weights)')
    for i in tqdm.tqdm(range(nre)):
        weight_sqrt = weights[:,i]
        weight_sqrt[np.isnan(weight_sqrt)] = 100.
        weight_sqrt[weight_sqrt < 0.005] = 0.005
        weight_sqrt = np.sqrt(weight_sqrt)  # use squre root of weight, to faciliate WLS, same as for phase.
        y2 = y[:,i]
        # y = np.copy(np.reshape(y2, (A.shape[0], -1)))
        y2 = np.expand_dims(y2, 1)
        # weight_sqrt = np.copy(weight_sqrt.reshape(A.shape[0], -1))
        weight_sqrt = np.expand_dims(weight_sqrt, 1)
        X, residual, ranks[i], _ = np.linalg.lstsq(np.multiply(A, weight_sqrt), np.multiply(y2, weight_sqrt),
                                                rcond=rcond)
        if residual.size > 0:
            residuals[:,i] = residual
        else:
            residuals[:,i] = np.squeeze(np.squeeze(A).dot(X))
        # ts_diff = np.squeeze(X * np.tile(tbase_diff, (1, num_pixel)))
        ts_diff = X[:,0] * tbase_diff[:,0]
        ts[1:, i] = np.cumsum(ts_diff)
    return ts, residuals, ranks


def linalg_weights3(A, y, weights, tbase_diff, num_pixel, rcond=1e-5):
    num_date = A.shape[1] + 1
    ts = np.zeros((num_date, nre), dtype=np.float32)
    residuals = np.zeros((A.shape[0], nre), dtype=np.float32)
    ranks = np.empty(nre, dtype=np.float32)
    ranks.fill(np.nan)

    #will do pixel-by-pixel inversion, because some pixels may not have data
    print('Run linear inversion on each pixel with weights')
    for i in tqdm.tqdm(range(nre)):
        weight_sqrt = weights[:,i]
        # weight_sqrt[np.isnan(weight_sqrt)] = 100.
        # weight_sqrt[weight_sqrt < 0.005] = 0.005
        # weight_sqrt = np.sqrt(weight_sqrt)  # use squre root of weight, to faciliate WLS, same as for phase.
        y2 = y[:,i]
        # y = np.copy(np.reshape(y2, (A.shape[0], -1)))
        y2 = np.expand_dims(y2, 1)
        # weight_sqrt = np.copy(weight_sqrt.reshape(A.shape[0], -1))
        weight_sqrt = np.expand_dims(weight_sqrt, 1)
        X, residual, ranks[i], _ = np.linalg.lstsq(np.multiply(A, weight_sqrt), np.multiply(y2, weight_sqrt),
                                                rcond=rcond)
        if residual.size > 0:
            residuals[:,i] = residual
        else:
            residuals[:,i] = np.squeeze(np.squeeze(A).dot(X))

        # ts_diff = np.squeeze(X * np.tile(tbase_diff, (1, num_pixel)))
        ts_diff = X[:,0] * tbase_diff[:,0]
        ts[1:, i] = np.cumsum(ts_diff)
    return ts, residuals, ranks


def linalg_weights4(A, y, weights, tbase_diff, num_pixel, rcond=1e-5):
    num_date = A.shape[1] + 1
    ts = np.zeros((num_date, nre), dtype=np.float32)
    residuals = np.zeros((A.shape[0], nre), dtype=np.float32)
    ranks = np.empty(nre, dtype=np.float32)
    ranks.fill(np.nan)

    #will do pixel-by-pixel inversion, because some pixels may not have data
    print('Run linear inversion on each pixel with squared weights')
    for i in tqdm.tqdm(range(nre)):
        weight_sqrt = weights[:,i]
        # weight_sqrt[np.isnan(weight_sqrt)] = 100.
        # weight_sqrt[weight_sqrt < 0.005] = 0.005
        weight_sqrt = weight_sqrt**2  # use square weights
        y2 = y[:,i]
        # y = np.copy(np.reshape(y2, (A.shape[0], -1)))
        y2 = np.expand_dims(y2, 1)
        # weight_sqrt = np.copy(weight_sqrt.reshape(A.shape[0], -1))
        weight_sqrt = np.expand_dims(weight_sqrt, 1)
        X, residual, ranks[i], _ = np.linalg.lstsq(np.multiply(A, weight_sqrt), np.multiply(y2, weight_sqrt),
                                                rcond=rcond)
        if residual.size > 0:
            residuals[:,i] = residual
        else:
            residuals[:,i] = np.squeeze(np.squeeze(A).dot(X))
        # ts_diff = np.squeeze(X * np.tile(tbase_diff, (1, num_pixel)))
        ts_diff = X[:,0] * tbase_diff[:,0]
        ts[1:, i] = np.cumsum(ts_diff)
    return ts, residuals, ranks


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
    num_ifgram = dx_stack.shape[0]
    nre = int(len(idxxy))
    dx_stack_masked = np.empty((num_ifgram, nre), dtype=np.float32)
    dx_stack_masked.fill(np.nan)
    conf_masked = np.empty((num_ifgram, nre), dtype=np.float32)
    conf_masked.fill(np.nan)

    # Could also do this via numba, but looks fast enough right now
    for i in tqdm.tqdm(range(dx_stack.shape[0])):
        dx_stack_masked[i,:] = dx_stack[i, :, :].ravel()[idxxy]
        conf_masked[i,:] = conf[i, :, :].ravel()[idxxy]

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

    ts_weights, residuals_weights, ranks_weights = linalg_weights(A, dx_stack_masked, conf_masked, tbase_diff, nre, rcond=1e-5)
    ts_weights2, residuals_weights2, ranks_weights2 = linalg_weights2(A, dx_stack_masked, conf_masked, tbase_diff, nre, rcond=1e-5)
    ts_weights3, residuals_weights3, ranks_weights3 = linalg_weights3(A, dx_stack_masked, conf_masked, tbase_diff, nre, rcond=1e-5)
    ts_weights4, residuals_weights4, ranks_weights4 = linalg_weights4(A, dx_stack_masked, conf_masked, tbase_diff, nre, rcond=1e-5)
    ts_noweights, residuals_noweights, ranks_noweights = linalg_noweights(A, dx_stack_masked, tbase_diff, nre, rcond=1e-5)

    tbase_diff2 = np.insert(tbase_diff, 0, 0)

    date_format = "%Y%m%d"

    date0_list = [str(i) for i in np.int32(date0_stack).tolist()]
    dates0 = np.array([dt.datetime.strptime(i, date_format) for i in date0_list])
    date1_list = [str(i) for i in np.int32(date1_stack).tolist()]
    dates1 = np.array([dt.datetime.strptime(i, date_format) for i in date1_list])
    ddates = dates1 - dates0
    ddates_day = np.array([i.days for i in ddates])

    date_list = [str(i) for i in np.int32(date0_stack).tolist()]
    img_dates = np.array([dt.datetime.strptime(i, date_format) for i in date_list])

    # residual vs. time span
    fig, ax = plt.subplots(1, 3, figsize=(18, 9), dpi=300)
    ax[0].plot(ddates_day, np.nanmean(residuals_weights, axis=1), 'k+', label='1 / weights')
    ax[0].plot(ddates_day, np.nanmean(residuals_weights2, axis=1), '+', color='darkred', label='sqrt(weights)')
    ax[0].plot(ddates_day, np.nanmean(residuals_weights3, axis=1), '+', color='magenta', label='weights')
    ax[0].set_title('Mean Residual from each pixel for each image pair (n=%d)'%residuals_weights.shape[0], fontsize=14)
    ax[0].set_xlabel('Time Difference [days]')
    ax[0].set_ylabel('residual')
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(img_dates, np.nanmean(residuals_weights, axis=1), 'k+', label='1 / weights')
    ax[1].plot(img_dates, np.nanmean(residuals_weights2, axis=1), '+', color='darkred', label='sqrt(weights)')
    ax[1].plot(img_dates, np.nanmean(residuals_weights3, axis=1), '+', color='magenta', label='weights')
    ax[1].plot(img_dates, np.nanmean(residuals_noweights, axis=1), '+', color='navy', label='weights')
    ax[1].set_title('Mean Residual for each image pair (n=%d)'%residuals_weights.shape[0], fontsize=14)
    ax[1].set_xlabel('First Image date')
    ax[1].set_ylabel('residual')
    ax[1].legend()
    ax[1].grid()

    # Generating Debugging and test plots
    fig, ax = plt.subplots(1, 3, figsize=(18, 9), dpi=300)
    ax[0].plot(np.cumsum(tbase_diff2), np.nanmean(ts_weights, axis=1), 'k-', label='1 / weights')
    ax[0].plot(np.cumsum(tbase_diff2), np.nanmean(ts_weights2, axis=1), '-', color='darkred', label='sqrt(weights)')
    ax[0].plot(np.cumsum(tbase_diff2), np.nanmean(ts_weights3, axis=1), '-', color='magenta', label='weights')
    ax[0].plot(np.cumsum(tbase_diff2), np.nanmean(ts_weights4, axis=1), '-', color='green', label='weights**2')
    ax[0].plot(np.cumsum(tbase_diff2), np.nanmean(ts_noweights, axis=1), '-', color='navy', label='no weights')
    ax[0].set_title('Mean offset (n=%d)'%nre, fontsize=14)
    ax[0].set_xlabel('Time [y]')
    ax[0].set_ylabel('Cumulative offset [pix]')
    ax[0].legend()
    ax[0].grid()

    for i in np.arange(0, ts.shape[1], 2000):
        if i == 0:
            ax[1].plot(np.cumsum(tbase_diff2), ts_weights[:,i], 'k-', lw=0.5, label='1 / weights')
            ax[1].plot(np.cumsum(tbase_diff2), ts_weights2[:,i], '-', color='darkred', lw=0.5, label='sqrt(weights)')
            ax[1].plot(np.cumsum(tbase_diff2), ts_noweights[:,i], '-', color='navy', lw=0.5, label='no weights')
        else:
            ax[1].plot(np.cumsum(tbase_diff2), ts_weights[:,i], 'k-', lw=0.5)
            ax[1].plot(np.cumsum(tbase_diff2), ts_weights2[:,i], '-', color='darkred', lw=0.5)
            ax[1].plot(np.cumsum(tbase_diff2), ts_noweights[:,i], '-', color='navy', lw=0.5)
    ax[1].set_xlabel('Time [y]')
    ax[1].set_ylabel('Cumulative offset [pix]')
    ax[1].set_title('Individual offsets (n=%d)'%nre, fontsize=14)
    ax[1].legend()

    # for i in np.arange(0, ts.shape[1], 1000):
    #     if i == 0:
    #         ax[2].plot(np.cumsum(tbase_diff), ts_weights[1:,i] - ts_noweights[1:,i], '-', color='navy', lw=1, label='weights - noweights')
    #         ax[2].plot(np.cumsum(tbase_diff), ts_weights[1:,i] - ts_weights2[1:,i], '-', color='darkred', lw=1, label='weights1 - weights2')
    #     else:
    #         ax[2].plot(np.cumsum(tbase_diff), ts_weights[1:,i] - ts_noweights[1:,i], '-', color='navy', lw=1)
    #         ax[2].plot(np.cumsum(tbase_diff), ts_weights[1:,i] - ts_weights2[1:,i], '-', color='darkred', lw=1)
    ax[2].plot(np.cumsum(tbase_diff2), np.nanmean(ts_weights, axis=1) - np.nanmean(ts_weights2, axis=1), '-', color='darkred', label='1 / weights - sqrt(weights)')
    ax[2].plot(np.cumsum(tbase_diff2), np.nanmean(ts_weights, axis=1) - np.nanmean(ts_weights3, axis=1), '-', color='magenta', label='1 / weights - weights')
    ax[2].plot(np.cumsum(tbase_diff2), np.nanmean(ts_weights, axis=1) - np.nanmean(ts_noweights, axis=1), '-', color='navy', label='1 / weights - no weights')
    ax[2].set_xlabel('Time [y]')
    ax[2].set_ylabel('$\Delta$ cumulative offset weights - noweights [pix]')
    ax[2].set_title('Mean $\Delta$ cumulative offsets weights - noweights (n=%d)'%nre, fontsize=14)
    ax[2].legend()
    ax[2].grid()

    fig.tight_layout()
    fig.savefig('Inverted_timeseries_weights_noweights.png', dpi=300)


    ## iterate through all dates and remove each date once - recreate design matrix every time
    # create a simple leave-one-out code (one date left out at a time)
    for i in range(len(unique_date)):
        if i == 0:
            idx = np.arange(0, len(unique_date), 1)
        elif i > 0 and i < len(unique_date):
            idx1 = np.arange(0, i, 1)
            idx2 = np.arange(i+1,len(unique_date), 1)
            idx = np.r_[idx1, idx2]

            # create boot-strapped dates
            unique_date_bs = np.take(unique_date, idx)
            # remove single date from list
            idx_stack0 = np.where(date0_stack == unique_date[i])[0].astype(np.int32)
            date0_stack_bs = np.delete(date0_stack, idx_stack0)
            date1_stack_bs = np.delete(date1_stack, idx_stack0)
            idx_stack1 = np.where(date1_stack_bs == unique_date[i])[0].astype(np.int32)
            date0_stack_bs = np.delete(date0_stack_bs, idx_stack1)
            date1_stack_bs = np.delete(date1_stack_bs, idx_stack1)

            num_date, tbase, date1s, date_list, date12_list, unique_date_bs = prepare_design_matrix_input(date0_stack_bs, date1_stack_bs, date_format = "%Y%m%d")
            num_ifgram = date0_stack_bs.shape[0]
            A, B, refDate = create_design_matrix(num_ifgram, num_date, tbase, date1s, date_list, refDate=None)
            tbase_diff = np.diff(tbase).reshape(-1, 1)

            # Unweighted Inversion for all pixels
            #remove boot-strapped date from weight and y matrix
            weight_sqrt =  np.delete(conf_masked, idx_stack0, axis=0)
            weight_sqrt =  np.delete(weight_sqrt, idx_stack1, axis=0)
            weight_sqrt[np.isnan(weight_sqrt)] = 100.
            weight_sqrt[weight_sqrt < 0.005] = 0.005
            weight_sqrt = 1. / weight_sqrt  # use squre root of weight, to faciliate WLS, same as for phase.
            y = np.delete(dx_stack_masked, idx_stack0, axis=0)
            y = np.delete(y, idx_stack1, axis=0)
            y = y.reshape(A.shape[0], -1)
            ts_weights, residuals_weights, ranks_weights = linalg_weights_bs(A, dx_stack_masked, conf_masked, idx_stack0, idx_stack1, tbase_diff, nre, rcond=1e-5)

            weight_sqrt = weight_sqrt.reshape(A.shape[0], -1)
            X, residuals, ranks, _ = np.linalg.lstsq(np.multiply(A, weight_sqrt), np.multiply(y, weight_sqrt), rcond=rcond)



#https://stackoverflow.com/questions/52452515/equivalent-to-numpy-linalg-lstsq-that-allows-weighting
import numpy as np
x, y = np.meshgrid(np.arange(0, 3), np.arange(0, 3))
x = x.ravel()
y = y.ravel()
values = np.sqrt(x+y+2)   # some values to fit
functions = np.stack([np.ones_like(y), y, x, x**2, y**2], axis=1)
functions = A
values = dx_stack_masked[:,0]
coeff_r = np.linalg.lstsq(functions, values, rcond=None)[0]
values_r = functions.dot(coeff_r)
weights = conf_masked[:,0]
coeff_r_weights = np.linalg.lstsq(functions*weights[:, None], values*weights, rcond=None)[0]
values_r_weights = functions.dot(coeff_r_weights)
plt.plot(values_r - values, 'k-', label='residuals no weights')
plt.plot(values_r_weights - values, '-', color='navy', label='residuals weights')
plt.legend()
plt.grid()
plt.show()

print(values_r - values)
