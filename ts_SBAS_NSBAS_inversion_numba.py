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
import correlation_confidence as cc

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import measure
from skimage.morphology import closing, disk
from osgeo import gdal
import pandas as pd
from numba import njit, prange
from numba_progress import ProgressBar
from scipy.signal import savgol_filter

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

    tbase = [i.days + i.seconds / (24 * 60 * 60) for i in (unique_dates - unique_dates[0])] #AM: why do you need to add seconds here? that will always be 0
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


#@njit(parallel=True)
def SBAS_noweights_numba(A, y, d_tbase, rcond=1e-10):
    #numba-based inversion with no weights
    num_date = A.shape[1] + 1
    n_ifg = y.shape[0]
    n_pt = y.shape[1]
    ts = np.zeros((num_date, n_pt), dtype=np.float32)
    residuals = np.empty(n_pt, dtype=np.float32)
    residuals.fill(np.nan)
    r_squared = np.empty(n_pt, dtype=np.float32)
    r_squared.fill(np.nan)
    residual_each_date = np.empty((n_ifg, n_pt), dtype=np.float32)
    residual_each_date.fill(np.nan)

    #will do pixel-by-pixel inversion, because some pixels may not have data
    for i in prange(n_pt):
        y2 = y[:,i].astype(np.float64)
        if np.any(np.isnan(y2)) or np.any(np.isinf(y2)):
            continue
        X, _, _, _ = np.linalg.lstsq(A.astype(np.float64), y2, rcond=rcond)
        # X, residual, ranks[i], _ = np.linalg.lstsq(A.astype(np.float64), y2, rcond=rcond)
        # for a time series with unconnected islands (rank > 1), there are no residuals returned
        # and you can obtain them via: np.linalg.norm(A.astype(np.float64) @ X - y2)**2
        X2 = X #* d_tbase
        ts[1:, i] = np.cumsum(X2).astype(np.float32)
        residuals[i] = np.linalg.norm(A.astype(np.float64) @ X - y2)**2
        residual = y2-np.dot(A.astype(np.float64), X)
        residual_each_date[:,i] = residual
        # residual_sumsq = np.nansum(residual**2)
        # residual_n = residual
        # residual_n[residual==0] = np.nan
        # residual_n[residual<0] = np.nan
        # residual_rms = np.sqrt(residual_sumsq/residual_n)
        # residual_rms_median[i] = np.nanmedian(residual_rms)
        #calculate median R2 for each point
        r_squared[i] = 1 - residuals[i] / np.sum((y2 - np.mean(y2))**2)
    return ts, residuals, r_squared, residual_each_date


#@njit(parallel=True)
def NSBAS_noweights_numba(G, y, tbase, gamma=1e-4, rcond=1e-10):
    # G    : Design matrix for incremental offset (1 between primary and secondary)
    n_ifg, n_pt = y.shape
    n_im = G.shape[1]+1
    ts = np.zeros((n_im, n_pt), dtype=np.float32)
    vel = np.zeros((n_pt), dtype=np.float32)
    vconst = np.zeros((n_pt), dtype=np.float32)
    residuals = np.empty(n_pt, dtype=np.float32)
    residuals.fill(np.nan)
    r_squared = np.empty(n_pt, dtype=np.float32)
    r_squared.fill(np.nan)
    residual_each_date = np.empty((n_ifg, n_pt), dtype=np.float32)
    residual_each_date.fill(np.nan)

    ### Set matrix of NSBAS part (bottom)
    Gbl = np.tril(np.ones((n_im, n_im-1), dtype=np.float32), k=-1) #lower tri matrix without diag
    # now add time constraints to link unconnected islands through tbase or dt_cumulative vector
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
                residuals[i] = residual
            else:
                # residuals[i] = Gall.astype(np.float64).dot(X)
                residuals[i] = np.linalg.norm(Gall @ X - y2)**2
        else:
            #currently not treating NaN in time series
            # y2 = np.concatenate((y[bool_pt_full, :], np.zeros((n_pt_full, n_im), dtype=np.float32)), axis=1).transpose()
            continue
        X2 = X[:n_im-1, :]
        # X2 = np.insert(X2, 0, 0) # adds zero to first (reference) date
        ts[1:, i] = np.cumsum(X2) # stores cumulative deformation
        vel[i] = X[n_im-1, :] # stores velocity slope
        vconst[i] = X[n_im, :] # stores constant velocity factor
        residual = Gall @ X - y2
        residual_each_date[:,i] = residual[:n_ifg,0]
        # residual_sumsq = np.nansum(residual**2, axis=0)
        # residual_n = residual
        # residual_n[residual==0] = np.nan
        # residual_n[residual<0] = np.nan
        # residual_rms = np.sqrt(residual_sumsq/residual_n)
        # residual_rms_median[i] = np.nanmedian(residual_rms)
        #calculate median R2 for each point
        r_squared[i] = 1 - residuals[i] / np.sum((y2 - np.mean(y2))**2)
    return ts, residuals, r_squared, residual_each_date, vel, vconst


def read_file(fn, b=1):
    ds = gdal.Open(fn)
    data = ds.GetRasterBand(b).ReadAsArray()
    ds = None
    return data

def ts_gaussian_sum_smooth(xdata, ydata, xeval, sigma, null_thresh=0.6):
    # https://stackoverflow.com/questions/24143320/gaussian-sum-filter-for-irregular-spaced-points
    """Apply gaussian sum filter to data.

    xdata, ydata : array
        Arrays of x- and y-coordinates of data.
        Must be 1d and have the same length.

    xeval : array
        Array of x-coordinates at which to evaluate the smoothed result

    sigma : float
        Standard deviation of the Gaussian to apply to each data point
        Larger values yield a smoother curve.

    null_thresh : float
        For evaluation points far from data points, the estimate will be
        based on very little data. If the total weight is below this threshold,
        return np.nan at this location. Zero means always return an estimate.
        The default of 0.6 corresponds to approximately one sigma away
        from the nearest datapoint.
    """
    # Distance between every combination of xdata and xeval
    # each row corresponds to a value in xeval
    # each col corresponds to a value in xdata
    delta_x = xeval[:, None] - xdata

    # Calculate weight of every value in delta_x using Gaussian
    # Maximum weight is 1.0 where delta_x is 0
    weights = np.exp(-0.5 * ((delta_x / sigma) ** 2))

    # Multiply each weight by every data point, and sum over data points
    smoothed = np.dot(weights, ydata)

    # Nullify the result when the total weight is below threshold
    # This happens at evaluation points far from any data
    # 1-sigma away from a data point has a weight of ~0.6
    nan_mask = weights.sum(1) < null_thresh
    smoothed[nan_mask] = np.nan

    # Normalize by dividing by the total weight at each evaluation point
    # Nullification above avoids divide by zero warning shere
    smoothed = smoothed / weights.sum(1)


    return smoothed


def ts_moving_average(data, window_size):
    window = np.ones(window_size) / window_size
    datae = np.empty(data.shape[0] + 2*int(np.ceil(window_size/2)))
    datae.fill(np.nan)
    #pad data with first and last value to avoid border effecs
    datae[0:int(np.ceil(window_size/2))] = data[0]
    datae[int(np.ceil(window_size/2)):int(np.ceil(window_size/2))+data.shape[0]] = data
    datae[-int(np.ceil(window_size/2)):] = data[-1]
    smoothed_data = np.convolve(datae, window, mode='same')
    smoothed_data = smoothed_data[int(np.ceil(window_size/2)):-int(np.ceil(window_size/2))]
    return smoothed_data

def get_landslide_loc(dx_stack, dy_stack, ddates, threshold_angle = 45, threshold_size = 10000, where = "all", res = 3, pad = 10):
    assert where in ["all", "centroid", "highest_val"], "Invalid option provided for where to put the landslide mask."
    directions = cc.calc_angle_numba(dx_stack, dy_stack) # returns angles in degree
    # std_dirs = cc.angle_variance(directions) # angle_variance scaled between 0 and 1
    print('Calculating std. dev. of angles through time')
    directions_sd = cc.nanstd_numba(directions)
    dbin = np.where(directions_sd < threshold_angle, 1, 0)
    labeled = measure.label(dbin, background=0, connectivity=2)
    info = measure.regionprops(labeled)
    # Filter connected components based on size
    filtered_labels = []
    for region in info:
        if region.area > threshold_size:
            filtered_labels.append(region.label)

    filtered_mask = np.isin(labeled, filtered_labels)

    #remove holes
    footprint = disk(5)
    closed = closing(filtered_mask, footprint)
    labeled = measure.label(closed, background=0, connectivity=2)

    if where == "all":
        slides = np.unique(labeled)
        slides = slides[slides > 0]
        print(f"Found {len(slides)} potential landslide(s).")
        masks = np.zeros((len(slides), dx_stack.shape[1], dx_stack.shape[2]))

        for i, slide in enumerate(slides):
            mask = np.zeros((dx_stack.shape[1], dx_stack.shape[2]))
            mask[labeled == slide] = 1
            masks[i,:,:] = mask
    else:
        # get centroid of landslide
        if where == "centroid":
            info = measure.regionprops(labeled)
            pts = [r.centroid for r in info]

        #get point with highest velocity
        elif where == "highest_vel":
            v = np.zeros(dx_stack.shape)
            for i in range(len(ddates)):
                v[i,:,:] = (np.sqrt((dx_stack[i]**2+dy_stack[i]**2))*res)/ddates[i].days*365

            v = cc.nanmean_numba(v)
            pts = []
            for label in np.unique(labeled):
                if label > 0:
                    temp = v.copy()
                    temp[labeled != label] = -9999
                    midx = np.argmax(temp)  # idx of flattened array
                    midx = np.unravel_index(midx, temp.shape)
                    pts.append(midx)
        masks = np.zeros((len(pts), dx_stack.shape[1], dx_stack.shape[2]))
        print(f"Found {len(pts)} potential landslide(s).")
        for i, pt in enumerate(pts):
            mask = np.zeros((dx_stack.shape[1], dx_stack.shape[2]))
            mask[int(pt[0])-pad:int(pt[0])+pad+1, int(pt[1])-pad:int(pt[1])+pad+1] = 1
            masks[i,:,:] = mask

    plt.figure()
    plt.imshow(directions_sd, vmin = 0, vmax = 90, cmap = "Reds_r")
    plt.colorbar()
    masks_sum = np.sum(masks, axis = 0)
    masks_sum[masks_sum == 0] = np.nan
    plt.imshow(masks_sum, alpha = 0.6, cmap = "Blues_r")
    
    return masks

def cmdLineParser():
    from argparse import RawTextHelpFormatter
    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--npy_out_path', default='npy', help='Output compressed numpy files', required=True)
    parser.add_argument('--area_name', help='Name of area of interest', required=True)
    parser.add_argument('--png_out_path', default='npy', help='Output PNG showing directional standard deviations, mask, and labels', required=False)
    parser.add_argument('--deltay_stack_scale', default=2., help='Output PNG showing directional standard deviations, mask, and labels', required=False)

    return parser.parse_args()


if __name__ == '__main__':
    #args = cmdLineParser()

    # Debugging:
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.png_out_path = 'png'
    args.area_name = "aoi9"
    args.npy_out_path = 'npy'
    args.png_out_path = 'png'

    #files = glob.glob(f"/raid-manaslu/amueting/PhD/Project3/PlanetScope_Data/{args.area_name}/all_scenes/disparity_maps/*L3B_polyfit-F.tif")
        
    files = glob.glob(f"/home/ariane/Documents/Project3/PlanetScope_Data/{args.area_name}/all_scenes/disparity_maps/*L3B_polyfit-F.tif")
    print(f"Found {len(files)} correlation pairs")
    #dem is just for plotting
    demname = f"/home/ariane/Documents/Project3/DEM_Data/CopernicusDEM_clip_{args.area_name}.tif"
    #mask_fname = "/home/ariane/Documents/Project3/PlanetScope_Data/aoi5/masks/aoi5_region1.npy.gz"

    # files = glob.glob("/raid/Planet_NWArg/PS2_aoi7/disparity_maps/*L3B_polyfit-F.tif")
    # mask_fname = "/raid/Planet_NWArg/PS2_aoi7/masks/aoi7_region1.npy.gz"
    # print(f"Found {len(files)} correlation pairs")


    if not os.path.exists(args.png_out_path):
        os.mkdir(args.png_out_path)

    # #Load masked file - either as Geotiff or as npy
    # print('Load mask data')
    # if os.path.exists(mask_fname):
    #     f = gzip.GzipFile(mask_fname, "r")
    #     mask = np.load(f)
    #     f = None

    # if os.path.exists(directions_sd_mask_geotiff_fname):
    #     ds = gdal.Open(directions_sd_mask_geotiff_fname)
    #     dxdy_size = ds.GetRasterBand(1).ReadAsArray().shape
    #     mask = ds.GetRasterBand(1).ReadAsArray()
    #     mask[mask == -9999] = np.nan
    #     gt = ds.GetGeoTransform()
    #     sr = ds.GetProjection()
    #     ds = None
    # elif os.path.exists(mask_fname):
    #     f = gzip.GzipFile(mask_fname, "r")
    #     mask = np.load(f)
    #     f = None
    # else:
    #     print('Could not find file')

    ### Load time series data stored in npy files
    bns = [os.path.basename(f) for f in files]
    dates0 = [dt.datetime.strptime(f[0:8], "%Y%m%d") for f in bns]
    dates1 = [dt.datetime.strptime(f.split("_")[3], "%Y%m%d") if len(f.split("_")[3]) == 8 else dt.datetime.strptime(f.split("_")[4], "%Y%m%d") for f in bns]
    print('Load dx data')
    # f = gzip.GzipFile(dx_npy_fname, "r")
    # dx_stack = np.load(f)
    # f = None
    dx_stack = np.asarray([read_file(f,1) for f in files])

    print('Load dy data')
    # f = gzip.GzipFile(dy_npy_fname, "r")
    # dy_stack = np.load(f)
    # f = None
    dy_stack = np.asarray([read_file(f,2) for f in files])

    dates0 = np.asarray(dates0)
    dates1 = np.asarray(dates1)
    ddates = dates1 - dates0
    ddates_day = np.array([i.days for i in ddates])

    print('Creating mask data')
    # masks = get_landslide_loc(dx_stack, dy_stack, ddates, pad = 20, where = "highest_vel", threshold_size = 5000)   
    masks = get_landslide_loc(dx_stack, dy_stack, ddates, pad = 20, where = "all", threshold_size = 5000, threshold_angle = 45)   
    
    #get mean vel (just for plotting)
    res = 3
    v = np.zeros(dx_stack.shape)
    for i in range(len(ddates)):
        v[i,:,:] = (np.sqrt((dx_stack[i]**2+dy_stack[i]**2))*res)/ddates[i].days*365

    v = cc.nanmean_numba(v)
    
    for idx in range(masks.shape[0]):
        mask = masks[idx,:,:]
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
        for i in range(dx_stack.shape[0]):
            dx_stack_masked[i,:] = dx_stack[i, :, :].ravel()[idxxy]
            dy_stack_masked[i,:] = dy_stack[i, :, :].ravel()[idxxy]

        #del dx_stack, dy_stack

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
        dx_ts_SBAS_noweights, dx_residuals_SBAS_noweights, dx_r2_SBAS_noweights, dx_residualdates_SBAS_noweights = SBAS_noweights_numba(A, dx_stack_masked, tbase_diff[:,0], rcond=1e-10)
        print('\t dy')
        dy_ts_SBAS_noweights, dy_residuals_SBAS_noweights, dy_r2_SBAS_noweights, dy_residualdates_SBAS_noweights = SBAS_noweights_numba(A, dy_stack_masked, tbase_diff[:,0], rcond=1e-10)
        # useful plot: length of correlation duration vs. rsquared
        # next step is to apply smoothing filter to time series
        # for a Gaussian filter, the time steps should be regular. You will first need to do a linear interpolation and then apply Gaussian filtering.
        print('\tSBAS: Median of all r2 from dx: %2.2f'%np.nanmedian(dx_r2_SBAS_noweights))
        print('\tSBAS: Median of all r2 from dy: %2.2f'%np.nanmedian(dy_r2_SBAS_noweights))
        print('\tSBAS: Median of all residuals from dx: %2.2f'%np.nanmedian(dx_residuals_SBAS_noweights))
        print('\tSBAS: Median of all residuals from dy: %2.2f'%np.nanmedian(dy_residuals_SBAS_noweights))
        print('\tSBAS: Sum of squared residuals from dx: %2.2f'%np.nansum(dx_residuals_SBAS_noweights**2))
        print('\tSBAS: Sum of squared residuals from dy: %2.2f'%np.nansum(dy_residuals_SBAS_noweights**2))
        # print('Median of all rms from dx: %2.2f'%np.nanmedian(dx_rms_SBAS_noweights))
        # print('Median of all rms from dy: %2.2f'%np.nanmedian(dy_rms_SBAS_noweights))

        ### Smooth time series
        # use simple moving average approach (will work with irregular data)
        window_size = 5
        dx_ts_SBAS_noweights_mv = np.empty_like(dx_ts_SBAS_noweights)
        dx_ts_SBAS_noweights_mv.fill(np.nan)
        for i in range(dx_ts_SBAS_noweights.shape[1]):
            dx_ts_SBAS_noweights_mv[:,i] = ts_moving_average(dx_ts_SBAS_noweights[:,i], window_size)

        dy_ts_SBAS_noweights_mv = np.empty_like(dy_ts_SBAS_noweights)
        dy_ts_SBAS_noweights_mv.fill(np.nan)
        for i in range(dy_ts_SBAS_noweights.shape[1]):
            dy_ts_SBAS_noweights_mv[:,i] = ts_moving_average(dy_ts_SBAS_noweights[:,i], window_size)

        ### Gaussian Sum filter with equal step size
        xeval = np.arange(np.min(np.cumsum(tbase_diff2)), np.ceil(np.max(np.cumsum(tbase_diff2))), 1/12) #create monthly spacing
        sigma = (1/12) * 1.5 # 1.5 months sigma
        dx_ts_SBAS_noweights_gss = np.empty((xeval.shape[0], dx_ts_SBAS_noweights.shape[1]))
        dx_ts_SBAS_noweights_gss.fill(np.nan)
        for i in range(dx_ts_SBAS_noweights.shape[1]):
            dx_ts_SBAS_noweights_gss[:,i] = ts_gaussian_sum_smooth(np.cumsum(tbase_diff2), dx_ts_SBAS_noweights[:,i], xeval, sigma, null_thresh=0.6)

        dy_ts_SBAS_noweights_gss = np.empty((xeval.shape[0], dy_ts_SBAS_noweights.shape[1]))
        dy_ts_SBAS_noweights_gss.fill(np.nan)
        for i in range(dy_ts_SBAS_noweights.shape[1]):
            dy_ts_SBAS_noweights_gss[:,i] = ts_gaussian_sum_smooth(np.cumsum(tbase_diff2), dy_ts_SBAS_noweights[:,i], xeval, sigma, null_thresh=0.6)

        # Linear interpolation
        dx_ts_SBAS_noweights_l = np.empty((xeval.shape[0], dx_ts_SBAS_noweights.shape[1]))
        dx_ts_SBAS_noweights_l.fill(np.nan)
        for i in range(dx_ts_SBAS_noweights.shape[1]):
            dx_ts_SBAS_noweights_l[:,i] = np.interp(xeval, xp=np.cumsum(tbase_diff2), fp=dx_ts_SBAS_noweights[:,i])

        dy_ts_SBAS_noweights_l = np.empty((xeval.shape[0], dy_ts_SBAS_noweights.shape[1]))
        dy_ts_SBAS_noweights_l.fill(np.nan)
        for i in range(dy_ts_SBAS_noweights.shape[1]):
            dy_ts_SBAS_noweights_l[:,i] = np.interp(xeval, xp=np.cumsum(tbase_diff2), fp=dy_ts_SBAS_noweights[:,i])

        ### Savitzky Golay filter - works only on regularly-spaced samples
        # window size must be larger than polynomial order
        window_size = 5
        polyorder = 3
        dx_ts_SBAS_noweights_sg = np.empty_like(dx_ts_SBAS_noweights_l)
        dx_ts_SBAS_noweights_sg.fill(np.nan)
        for i in range(dx_ts_SBAS_noweights.shape[1]):
            dx_ts_SBAS_noweights_sg[:,i] = savgol_filter(dx_ts_SBAS_noweights_l[:,i], window_size, polyorder, mode='nearest')

        dy_ts_SBAS_noweights_sg = np.empty_like(dy_ts_SBAS_noweights_l)
        dy_ts_SBAS_noweights_sg.fill(np.nan)
        for i in range(dy_ts_SBAS_noweights.shape[1]):
            dy_ts_SBAS_noweights_sg[:,i] = savgol_filter(dy_ts_SBAS_noweights_l[:,i], window_size, polyorder, mode='nearest')

        # NSBAS - no weights
        print('\nRun linear NSBAS inversion on each pixel with no weights')
        print('\t dx')
        dx_ts_NSBAS_noweights, dx_residuals_NSBAS_noweights, dx_r2_NSBAS_noweights, dx_residualdates_NSBAS_noweights, dx_ts_NSBAS_noweights_vel, dx_ts_NSBAS_noweights_vconst = NSBAS_noweights_numba(A, dx_stack_masked, tbase, rcond=1e-10)
        print('\t dy')
        dy_ts_NSBAS_noweights, dy_residuals_NSBAS_noweights, dy_r2_NSBAS_noweights, dy_residualdates_NSBAS_noweights, dy_ts_NSBAS_noweights_vel, dy_ts_NSBAS_noweights_vconst = NSBAS_noweights_numba(A, dy_stack_masked, tbase, rcond=1e-10)
        print('\tNSBAS: Median of all r2 from dx: %2.2f'%np.nanmedian(dx_r2_NSBAS_noweights))
        print('\tNSBAS: Median of all r2 from dy: %2.2f'%np.nanmedian(dy_r2_NSBAS_noweights))
        print('\tNSBAS: Median of all residuals from dx: %2.2f'%np.nanmedian(dx_residuals_NSBAS_noweights))
        print('\tNSBAS: Median of all residuals from dy: %2.2f'%np.nanmedian(dy_residuals_NSBAS_noweights))
        print('\tNSBAS: Sum of squared residuals from dx: %2.2f'%np.nansum(dx_residuals_NSBAS_noweights**2))
        print('\tNSBAS: Sum of squared residuals from dy: %2.2f'%np.nansum(dy_residuals_NSBAS_noweights**2))

        fig, ax = plt.subplots(2, 2, figsize=(12,5))
        ax[0,0].plot(np.cumsum(tbase_diff2), np.nanmean(dx_ts_NSBAS_noweights, axis=1), '-', color='darkblue', label='NSBAS')
        ax[0,0].plot(np.cumsum(tbase_diff2), np.nanmean(dx_ts_SBAS_noweights, axis=1), '-', color='firebrick', label='SBAS')
        ax[0,0].set_title('Unsmoothed mean dx offset (n=%d)'%nre, fontsize=14)
        ax[0,0].set_xlabel('Time [y]')
        ax[0,0].set_ylabel('Cumulative dx offset [pix]')
        ax[0,0].legend()
        ax[0,0].grid()
        ax[0,1].plot(np.cumsum(tbase_diff2), np.nanmean(dy_ts_NSBAS_noweights, axis=1), '-', color='darkblue', label='NSBAS')
        ax[0,1].plot(np.cumsum(tbase_diff2), np.nanmean(dy_ts_SBAS_noweights, axis=1), '-', color='firebrick', label='SBAS')
        ax[0,1].set_title('Unsmoothed mean dy offset (n=%d)'%nre, fontsize=14)
        ax[0,1].set_xlabel('Time [y]')
        ax[0,1].set_ylabel('Cumulative dy offset [pix]')
        ax[0,1].legend()
        ax[0,1].grid()
        ax[1,0].plot(np.cumsum(tbase_diff2), np.nanmean(dx_ts_NSBAS_noweights, axis=1), '-', lw=0.5, color='darkblue', label='NSBAS')
        ax[1,0].plot(np.cumsum(tbase_diff2), np.nanmean(dx_ts_SBAS_noweights, axis=1), '-', lw=0.5, color='firebrick', label='SBAS')
        ax[1,0].plot(np.cumsum(tbase_diff2), np.nanmean(dx_ts_SBAS_noweights_mv, axis=1), '-', lw=1, color='firebrick', label='SBAS moving average')
        ax[1,0].plot(xeval, np.nanmean(dx_ts_SBAS_noweights_gss, axis=1), '-o', ms=2, lw=1, color='firebrick', label='SBAS gaussian smoothing')
        ax[1,0].plot(xeval, np.nanmean(dx_ts_SBAS_noweights_sg, axis=1), '-x', ms=2, lw=1, color='pink', label='SBAS Savitzky-Golay')
        ax[1,0].set_title('Smoothed Mean dx offset (n=%d)'%nre, fontsize=14)
        ax[1,0].set_xlabel('Time [y]')
        ax[1,0].set_ylabel('Cumulative dx offset [pix]')
        ax[1,0].legend()
        ax[1,0].grid()
        ax[1,1].plot(np.cumsum(tbase_diff2), np.nanmean(dy_ts_NSBAS_noweights, axis=1), '-', lw=0.5, color='darkblue', label='NSBAS')
        ax[1,1].plot(np.cumsum(tbase_diff2), np.nanmean(dy_ts_SBAS_noweights, axis=1), '-', lw=0.5, color='firebrick', label='SBAS')
        ax[1,1].plot(np.cumsum(tbase_diff2), np.nanmean(dy_ts_SBAS_noweights_mv, axis=1), '-', lw=1, color='firebrick', label='SBAS moving average')
        ax[1,1].plot(xeval, np.nanmean(dy_ts_SBAS_noweights_gss, axis=1), '-o', ms=2, lw=1, color='firebrick', label='SBAS gaussian smoothing')
        ax[1,1].plot(xeval, np.nanmean(dy_ts_SBAS_noweights_sg, axis=1), '-x', ms=2, lw=1, color='pink', label='SBAS Savitzky-Golay')
        ax[1,1].set_title('Smoothed Mean dy offset (n=%d)'%nre, fontsize=14)
        ax[1,1].set_xlabel('Time [y]')
        ax[1,1].set_ylabel('Cumulative dy offset [pix]')
        ax[1,1].legend()
        ax[1,1].grid()
        fig.tight_layout()
        fig.savefig(os.path.join(args.png_out_path, f'{args.area_name}_dx_dy_SBAS_NSBAS_inversion_region{idx}.png'), dpi=300)
        # fig.savefig(os.path.join(args.png_out_path, f'{args.area_name}_dx_dy_SBAS_NSBAS_inversion_comparison.png'), dpi=300)

     
        ## Create map view of r2 from residual estimation for every pixel
        # take r2 values for all masked pixels and turn into map view
        dx_r2_SBAS_noweights_map = np.zeros_like(mask, dtype=np.float32)
        dx_r2_SBAS_noweights_map.fill(np.nan)
        dx_r2_SBAS_noweights_map.ravel()[idxxy] = dx_r2_SBAS_noweights

        dy_r2_SBAS_noweights_map = np.zeros_like(mask, dtype=np.float32)
        dy_r2_SBAS_noweights_map.fill(np.nan)
        dy_r2_SBAS_noweights_map.ravel()[idxxy] = dy_r2_SBAS_noweights

        dx_r2_NSBAS_noweights_map = np.zeros_like(mask, dtype=np.float32)
        dx_r2_NSBAS_noweights_map.fill(np.nan)
        dx_r2_NSBAS_noweights_map.ravel()[idxxy] = dx_r2_NSBAS_noweights

        dy_r2_NSBAS_noweights_map = np.zeros_like(mask, dtype=np.float32)
        dy_r2_NSBAS_noweights_map.fill(np.nan)
        dy_r2_NSBAS_noweights_map.ravel()[idxxy] = dy_r2_NSBAS_noweights

        fig, ax = plt.subplots(2, 2, figsize=(12,8))
        vmin = 0.7
        vmax = 1
        im0 = ax[0,0].imshow(dx_r2_SBAS_noweights_map, cmap='viridis', vmin=vmin, vmax=vmax)
        ax[0,0].set_title('R2 of SBAS dx', fontsize=14)
        im1 = ax[0,1].imshow(dy_r2_SBAS_noweights_map, cmap='viridis', vmin=vmin, vmax=vmax)
        ax[0,1].set_title('R2 of SBAS dy', fontsize=14)
        im2 = ax[1,0].imshow(dx_r2_NSBAS_noweights_map, cmap='viridis', vmin=vmin, vmax=vmax)
        ax[1,0].set_title('R2 of NSBAS dx', fontsize=14)
        im3 = ax[1,1].imshow(dy_r2_NSBAS_noweights_map, cmap='viridis', vmin=vmin, vmax=vmax)
        ax[1,1].set_title('R2 of NSBAS dy', fontsize=14)
        # fig.colorbar(im0, ax=ax.ravel().tolist())
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im0, cax=cbar_ax)
        # fig.tight_layout()
        fig.savefig(os.path.join(args.png_out_path, f'{args.area_name}_dx_dy_SBAS_NSBAS_r2_mapview_region{idx}.png'), dpi=300)

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
        
        
        #map plotting
        
        cmd = f"gdaldem hillshade {demname} {demname[:-4]}_HS.tif"
        os.system(cmd)
        
        hs = read_file(f"{demname[:-4]}_HS.tif")
        vplot = v.copy()
        vplot[mask == 0] = np.nan
        unique_dates = np.union1d(np.unique(dates0), np.unique(dates1))
        xeval_dates = [min(unique_dates) + dt.timedelta(days = x*365.25) for x in xeval]
        
        fig, ax = plt.subplots(1,3, figsize = (18,5))     
        ax[0].imshow(hs, cmap = "Greys_r")
        p = ax[0].imshow(vplot, cmap = "Reds", vmin = 0, vmax = 10, alpha = 0.8)
        plt.colorbar(p, ax = ax[0], label = "Velocity [m/yr]")
        ax[1].axhline(c = "gray", ls = "--")
        ax[1].plot(unique_dates, np.nanmean(dx_ts_SBAS_noweights, axis=1)*res, '-', lw=1, color='firebrick', label='SBAS')
        ax[1].plot(xeval_dates, np.nanmean(dx_ts_SBAS_noweights_sg, axis=1)*res, '-x', ms=2, lw=1, color='indigo', label='SBAS Savitzky-Golay')
        ax[1].grid()
        ax[1].set_ylim(-5,25)
        ax[1].set_ylabel("EW Displacement [m]")
        ax[1].set_xlabel("Time")
        ax[1].legend()
        ax[2].axhline(c = "gray", ls = "--")
        ax[2].plot(unique_dates, np.nanmean(dy_ts_SBAS_noweights, axis=1)*res, '-', lw=1, color='firebrick', label='SBAS')
        ax[2].plot(xeval_dates, np.nanmean(dy_ts_SBAS_noweights_sg, axis=1)*res, '-x', ms=2, lw=1.2, color='indigo', label='SBAS Savitzky-Golay')
        ax[2].grid()
        ax[2].set_ylim(-5,25)
        ax[2].set_ylabel("NS Displacement [m]")
        ax[2].set_xlabel("Time")
        ax[2].legend()
        plt.suptitle(args.area_name)
        plt.tight_layout()
        plt.savefig(os.path.join(args.png_out_path, f'{args.area_name}_dx_dy_SBAS_mapview_region{idx}.png'), dpi=300)


        