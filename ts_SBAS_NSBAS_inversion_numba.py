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

from osgeo import gdal

from numba import njit, prange
from numba_progress import ProgressBar

DESCRIPTION = """
Run SBAS or NSBAS time series inversion on offset pixels via numpy and numba. Takes advantage of multiple cores, but requires memory. Very fast, but only useful for limited number of points (up to 1e5) and limited timesteps (up to 100).
This reads in the offset timeseries and a landslide mask file (e.g., created with generate_landslide_mask.py) and an uncertainty offset file (IQR, generated with create_offset_confidence.py and --method 2).
"""

EXAMPLE = """example:
ts_inversion_numba.py \
    --area_name aoi7 \
    --npy_out_path npy \
    --png_out_path npy
"""

def create_design_matrix_cumulative_displacement(num_ifgram, dates0, dates1):
    # create design matrix (usually called G or J):
    # For a matrix with -1 at primary and 1 at secondary. (n_ifg, n_im):  Unknown is cumulative displacement.
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


def create_design_matrix_incremental_displacement(num_ifgram, dates0, dates1):
    # create design matrix (usually called G or J):
    # For a matrix with -1 at primary and 1 at secondary. (n_ifg, n_im):  Unknown is cumulative displacement.
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
        A[i, ind1:ind2] = 1

    # Remove reference date as it can not be resolved
    ref_date = dt.datetime.strftime(min(dates0),"%Y%m%d")

    ind_r = date_list.index(ref_date)
    A = np.hstack((A[:, 0:ind_r], A[:, (ind_r+1):]))
    return A, ref_date, tbase


@njit(parallel=True)
def SBAS_noweights_numba(A, y, tbase_diff, num_pixel, rcond=1e-10):
    #numba-based inversion with no weights
    num_date = A.shape[1] + 1
    ts = np.zeros((num_date, nre), dtype=np.float32)
    residuals = np.empty((A.shape[0], nre), dtype=np.float32)
    residuals.fill(np.nan)
    # ranks = np.empty(nre, dtype=np.float32)
    # ranks.fill(np.nan)

    #will do pixel-by-pixel inversion, because some pixels may not have data
    for i in prange(nre):
        y2 = y[:,i].astype(np.float64)
        if np.any(np.isnan(y2)) or np.any(np.isinf(y2)):
            continue
        X, residual, _, _ = np.linalg.lstsq(A.astype(np.float64), y2, rcond=rcond)
        # X, residual, ranks[i], _ = np.linalg.lstsq(A.astype(np.float64), y2, rcond=rcond)
        if residual.size > 0:
            residuals[:,i] = residual
        else:
            residuals[:,i] = A.astype(np.float64).dot(X)
        ts[1:, i] = np.cumsum(X)
    return ts, residuals#, ranks


#@njit(parallel=True)
def NSBAS_noweights_numba(G, y, tbase, gamma=1e-4, rcond=1e-10):
    # G    : Design matrix for cumulative offset (-1 at primary and 1 at secondary)
    # If the design matrix has 1s between primary and secondary, the unknown you are solving for is incremental displacement.
    n_ifg, n_pt = y.shape
    n_im = G.shape[1]+1
    cum_def = np.zeros((n_im, n_pt), dtype=np.float32)*np.nan
    vel = np.zeros((n_pt), dtype=np.float32)*np.nan
    vconst = np.zeros((n_pt), dtype=np.float32)*np.nan
    residuals = np.empty((n_ifg, n_pt), dtype=np.float32)
    residuals.fill(np.nan)

    ### Set matrix of NSBAS part (bottom)
    #has dimensions of number of unique dates and number of unique dates - 1
    Gbl = np.tril(np.ones((n_im, n_im-1), dtype=np.float32), k=-1) #lower tri matrix without diag
    # now add time constraints through tbase or dt_cumulative vector
    Gbr = -np.ones((n_im, 2), dtype=np.float32)
    Gbr[:, 0] = -tbase
    Gb = np.concatenate((Gbl, Gbr), axis=1)*gamma

    #combine connectivity matrix with lower triangle - will add constraints for the inversion
    Gt = np.concatenate((G, np.zeros((n_ifg, 2), dtype=np.float32)), axis=1)
    Gt = np.concatenate((G, np.ones((n_ifg, 2), dtype=np.float32)), axis=1)
    Gall = np.concatenate((Gt, Gb)).astype(np.float64)

    for i in prange(n_pt):
        y2 = y[:,i].astype(np.float64)
        y2 = np.expand_dims(y2, axis=0)
        #test if there are any NaN in y/correlation time series
        bool_pt_full = np.all(~np.isnan(y2), axis=1)
        n_pt_full = bool_pt_full.sum()
        y2 = np.concatenate((y2[bool_pt_full, :], np.zeros((n_pt_full, n_im), dtype=np.float32)), axis=1).transpose()
        if bool_pt_full == True:
            #will use all points
            X, residual, _, _ = np.linalg.lstsq(Gall, y2, rcond=rcond)
            if residual.size > 0:
                residuals[:,i] = residual
            else:
                residuals[:,i] = Gall.astype(np.float64).dot(X)

        else:
            #currently not treating NaN in time series
            # y2 = np.concatenate((y[bool_pt_full, :], np.zeros((n_pt_full, n_im), dtype=np.float32)), axis=1).transpose()
            continue
        X2 = X[:n_im-1, :]
        X2 = np.insert(X2, 0, 0) # adds zero to first (reference) date
        cum_def[:, i] = np.cumsum(X2) # stores cumulative deformation
        vel[i] = X[n_im-1, :] # stores velocity slope
        vconst[i] = X[n_im, :] # stores constant velocity factor
    return cum_def, residuals, vel, vconst


def read_file(fn, b=1):
    ds = gdal.Open(fn)
    data = ds.GetRasterBand(b).ReadAsArray()
    ds = None
    return data


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

    files = glob.glob("/raid-manaslu/amueting/PhD/Project3/PlanetScope_Data/aoi7/all_scenes/disparity_maps/*L3B_polyfit-F.tif")
    mask_fn = "/raid-manaslu/amueting/PhD/Project3/PlanetScope_Data/aoi7/masks/aoi7_region1.npy.gz"
    bns = [os.path.basename(f) for f in files]
    dx_stack = np.asarray([read_file(f,1) for f in files])
    dy_stack = np.asarray([read_file(f,2) for f in files])
    dates0 = [dt.datetime.strptime(f[0:8], "%Y%m%d") for f in bns]
    dates1 = [dt.datetime.strptime(f.split("_")[3], "%Y%m%d") if len(f.split("_")[3]) == 8 else dt.datetime.strptime(f.split("_")[4], "%Y%m%d") for f in bns]

    area_name = os.path.join(args.npy_out_path, args.area_name)
    deltay_stack_scale = args.deltay_stack_scale

    if not os.path.exists(args.png_out_path):
        os.mkdir(args.png_out_path)


    # mask_fname = 'masks/aoi7_region1.npy.gz'
    directions_sd_mask_npy_fname = area_name + '_directions_sd_mask.npy.gz'
    directions_sd_mask_geotiff_fname = area_name + '_directions_sd_mask.tif'
    date0_stack_fname = area_name + "_date0.npy.gz"
    date1_stack_fname = area_name + "_date1.npy.gz"
    deltay_stack_fname = area_name + "_deltay.npy.gz"
    dx_npy_fname = area_name + "_dx.npy.gz"
    dy_npy_fname = area_name + "_dy.npy.gz"

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
    elif os.path.exists(mask_fname):
        f = gzip.GzipFile(mask_fname, "r")
        mask = np.load(f)
        f = None
    else:
        print('Could not find file')

    ### Load time series data stored in npy files
    print('Load dx data')
    f = gzip.GzipFile(dx_npy_fname, "r")
    dx_stack = np.load(f)
    f = None

    print('Load dy data')
    f = gzip.GzipFile(dy_npy_fname, "r")
    dy_stack = np.load(f)
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
    A, ref_date, tbase = create_design_matrix_incremental_displacement(num_ifgram, dates0, dates1)
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
    dx_ts_SBAS_noweights_numba, dx_residuals_SBAS_noweights_numba = SBAS_noweights_numba(A, dx_stack_masked, tbase_diff, nre, rcond=1e-10)
    print('\t dy')
    dy_ts_SBAS_noweights_numba, dy_residuals_SBAS_noweights_numba= SBAS_noweights_numba(A, dy_stack_masked, tbase_diff, nre, rcond=1e-10)

    # NSBAS - no weights
    print('\nRun linear NSBAS inversion on each pixel with no weights')
    print('\t dx')
    dx_ts_NSBAS_noweights_numba, dx_residuals_NSBAS_noweights_numba, dx_ts_NSBAS_noweights_vel, dx_ts_NSBAS_noweights_vconst = NSBAS_noweights_numba(A, dx_stack_masked, tbase, rcond=1e-10)
    print('\t dy')
    dy_ts_NSBAS_noweights_numba, dy_residuals_NSBAS_noweights_numba, dy_ts_NSBAS_noweights_vel, dy_ts_NSBAS_noweights_vconst = NSBAS_noweights_numba(A, dy_stack_masked, tbase, rcond=1e-10)

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

    fig, ax = plt.subplots(2, 2, figsize=(12,5))
    ax[0,0].plot(np.cumsum(tbase_diff2), np.nanmean(dx_ts_NSBAS_noweights_numba, axis=1), '-', color='darkblue', label='NSBAS')
    ax[0,0].plot(np.cumsum(tbase_diff2), np.nanmean(dx_ts_SBAS_noweights_numba, axis=1), '-', color='firebrick', label='SBAS')
    ax[0,0].set_title('Mean dx offset (n=%d)'%nre, fontsize=14)
    ax[0,0].set_xlabel('Time [y]')
    ax[0,0].set_ylabel('Cumulative dx offset [pix]')
    ax[0,0].legend()
    ax[0,0].grid()
    ax[0,1].plot(np.cumsum(tbase_diff2), np.nanmean(dy_ts_NSBAS_noweights_numba, axis=1), '-', color='darkblue', label='NSBAS')
    ax[0,1].plot(np.cumsum(tbase_diff2), np.nanmean(dy_ts_SBAS_noweights_numba, axis=1), '-', color='firebrick', label='SBAS')
    ax[0,1].set_title('Mean dy offset (n=%d)'%nre, fontsize=14)
    ax[0,1].set_xlabel('Time [y]')
    ax[0,1].set_ylabel('Cumulative dy offset [pix]')
    ax[0,1].legend()
    ax[0,1].grid()
    ax[1,0].plot(dates0, np.nanmean(dx_residuals_NSBAS_noweights_numba, axis=1), 'o', color='darkblue', label='NSBAS')
    ax[1,0].plot(dates0, np.nanmean(dx_residuals_SBAS_noweights_numba, axis=1), '+', color='firebrick', label='SBAS')
    ax[1,0].set_title('dx residuals (n=%d)'%nre, fontsize=14)
    ax[1,0].set_xlabel('Starting date of correlation pair')
    ax[1,0].set_ylabel('Mean Residual [pix]')
    ax[1,0].legend()
    ax[1,0].grid()
    ax[1,1].plot(dates0, np.nanmean(dy_residuals_NSBAS_noweights_numba, axis=1), 'o', color='darkblue', label='NSBAS')
    ax[1,1].plot(dates0, np.nanmean(dy_residuals_SBAS_noweights_numba, axis=1), '+', color='firebrick', label='SBAS')
    ax[1,1].set_title('dy residuals (n=%d)'%nre, fontsize=14)
    ax[1,1].set_xlabel('Starting date of correlation pair')
    ax[1,1].set_ylabel('Mean Residual [pix]')
    ax[1,1].legend()
    ax[1,1].grid()
    fig.tight_layout()
    fig.savefig(os.path.join(png_out_path, '%s_dx_dy_SBAS_NSBAS_residuals_inversion.png'%area_name), dpi=300)

    # # Export inverted ts to npy files
    # if os.path.exists(dx_ts_tweights_numba_fn) is False:
    #     f = gzip.GzipFile(dx_ts_tweights_numba_fn, "w")
    #     np.save(file=f, arr=dx_ts_tweights_numba)
    #     f.close()
    #     f = None
    #
    # if os.path.exists(dy_ts_tweights_numba_fn) is False:
    #     f = gzip.GzipFile(dy_ts_tweights_numba_fn, "w")
    #     np.save(file=f, arr=dy_ts_tweights_numba)
    #     f.close()
    #     f = None
    #
    #
    # if os.path.exists(dx_ts_rweights_numba_fn) is False:
    #     f = gzip.GzipFile(dx_ts_rweights_numba_fn, "w")
    #     np.save(file=f, arr=dx_ts_rweights_numba)
    #     f.close()
    #     f = None
    #
    # if os.path.exists(dy_ts_rweights_numba_fn) is False:
    #     f = gzip.GzipFile(dy_ts_rweights_numba_fn, "w")
    #     np.save(file=f, arr=dy_ts_rweights_numba)
    #     f.close()
    #     f = None
    #
    # if os.path.exists(dx_ts_rweights2_numba_fn) is False:
    #     f = gzip.GzipFile(dx_ts_rweights2_numba_fn, "w")
    #     np.save(file=f, arr=dx_ts_rweights2_numba)
    #     f.close()
    #     f = None
    #
    # if os.path.exists(dy_ts_rweights2_numba_fn) is False:
    #     f = gzip.GzipFile(dy_ts_rweights2_numba_fn, "w")
    #     np.save(file=f, arr=dy_ts_rweights2_numba)
    #     f.close()
    #     f = None
