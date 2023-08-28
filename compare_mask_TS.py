#!/usr/bin/env python
"""
@author: bodo bookhagen, bodo.bookhagen@uni-potsdam.de
"""
import warnings, argparse, os, tqdm, datetime, gzip
import numpy as np
from numba import njit, prange

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LogNorm
import matplotlib.dates as mdates
from mintpy.utils import arg_utils, readfile, utils as ut, plot as pp
from mintpy.defaults.plot import *

import pandas as pd
# from pandas.plotting import register_matplotlib_converters
# register_matplotlib_converters()

EXAMPLE = """example:
compare_mask_TS.py \
    --az_ts1_file timeseriesAz_var.h5 \
    --rg_ts1_file timeseriesRg_var.h5 \
    --az_ts2_file timeseriesAz_no.h5 \
    --rg_ts2_file timeseriesRg_no.h5 \
    --mask_file aoi4_var_velocity_mask.h5 \
    --out1_pngfname aoi4_compare_var_no_ts.png \
    --out2_pngfname aoi4_compare_var_no_ts_velocity.png
"""

DESCRIPTION = """
Extract time series for masked region from dx (range) and dy (azimuth). Uses two time series as input and compares them.
Mask dataset need to be same size as TS. Save clipped dataset to npy file and create plot.

Aug-2023, Bodo Bookhagen (bodo.bookhagen@uni-potsdam.de) and Ariane Mueting (mueting@uni-potsdam.de)
"""

def cmdLineParser():
    from argparse import RawTextHelpFormatter
    parser = argparse.ArgumentParser(description=DESCRIPTION, epilog=EXAMPLE, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--rg_ts1_file', help='dx (Rg) offset timeseries 1', required=True)
    parser.add_argument('--az_ts1_file', help='dy (Az) offset timeseries 1', required=True)
    parser.add_argument('--rg_ts2_file', help='dx (Rg) offset timeseries 2', required=True)
    parser.add_argument('--az_ts2_file', help='dy (Az) offset timeseries 2', required=True)
    parser.add_argument('--rg_rs1_file', default='', help='dx (Rg) offset timeseries 1 - residual', required=False)
    parser.add_argument('--az_rs1_file', default='',help='dy (Az) offset timeseries 1 - residual', required=False)
    parser.add_argument('--rg_rs2_file', default='',help='dx (Rg) offset timeseries 2 - residual', required=False)
    parser.add_argument('--az_rs2_file', default='',help='dy (Az) offset timeseries 2 - residual', required=False)
    parser.add_argument('--prcp_threshold', type=np.int8, default='90', help='Percentile threshold (0-100) for residual masking', required=False)
    parser.add_argument('--mask_file', help='Mask file with data == 1. Allowed are npy, npy.gz, h5, and tif files.', required=True)
    parser.add_argument('--out1_pngfname', default="", help='Output TS plot in PNG format', required=True)
    parser.add_argument('--out2_pngfname', default="", help='Output TS plot in PNG format', required=True)
    return parser.parse_args()


@njit(parallel=True)
def dxdy_linear_interpolation(tsamples_monthly, ts1_delta_day_ar, rg_data_cum, az_data_cum, nre):
    az_data_cum_li = np.empty((tsamples_monthly.shape[0], nre), dtype=np.float32)
    az_data_cum_li.fill(np.nan)
    rg_data_cum_li = np.empty((tsamples_monthly.shape[0], nre), dtype=np.float32)
    rg_data_cum_li.fill(np.nan)
    for i in prange(rg_data_cum.shape[1]):
        #iterate through each pixel in mask and perform linear interpolation for dx and dy
        fit_dx = np.interp(x=tsamples_monthly, xp=np.cumsum(ts1_delta_day_ar), fp=rg_data_cum[:,i]) #using numpy - it's faster with numba
        fit_dy = np.interp(x=tsamples_monthly, xp=np.cumsum(ts1_delta_day_ar), fp=az_data_cum[:,i]) #using numpy - it's faster with numba
        az_data_cum_li[:,i] =  fit_dy
        rg_data_cum_li[:,i] =  fit_dx
    return az_data_cum_li, rg_data_cum_li


if __name__ == '__main__':
    np.seterr(divide='ignore', invalid='ignore')
    warnings.simplefilter("ignore")

    args = cmdLineParser()

    #testing purposes:
    # parser = argparse.ArgumentParser(description='Extract time series from mask (mask needs to be same size as TS).')
    # args = parser.parse_args()
    # args.az_ts1_file = 'timeseriesAz_var.h5'
    # args.rg_ts1_file = 'timeseriesRg_var.h5'
    # args.az_ts2_file = 'timeseriesAz_no.h5'
    # args.rg_ts2_file = 'timeseriesRg_no.h5'
    # args.az_rs1_file = ''
    # args.rg_rs1_file = ''
    # args.az_rs2_file = ''
    # args.rg_rs2_file = ''
    # args.az_rs1_file = 'residualInvAz_var.h5'
    # args.rg_rs1_file = 'residualInvRg_var.h5'
    # args.az_rs2_file = 'residualInvAz_no.h5'
    # args.rg_rs2_file = 'residualInvRg_no.h5'
    # args.mask_file = 'aoi4_var_velocity_mask.h5'
    # args.out1_pngfname = 'aoi4_compare_var_no_ts.png'
    # args.out2_pngfname = 'aoi4_compare_var_no_ts.png'
    # args.prcp_threshold = 90

    args.atr = readfile.read_attribute(args.az_ts1_file)
    args.coord = ut.coordinate(args.atr)

    # input file info
    ts1 = ut.timeseries(file=args.az_ts1_file)
    dates1 = ts1.get_date_list()
    ts1=None

    ts2 = ut.timeseries(file=args.az_ts2_file)
    dates2 = ts2.get_date_list()
    Copernicus_DSM_HMA_90m_int16_epsg32644=None

    # Both time series contain cumulative offset. Will need to convert to velocity before analysis
    print('Reading Rg timeseries 1... ',end='',flush=True)
    rg1_data, rg1_atr = readfile.read(args.rg_ts1_file, datasetName='timeseries')
    rg1_data -= rg1_data[0] #rg_data[0] should be all 0
    rg1_data *= float(rg1_atr['X_STEP'])
    print('done ')

    print('Reading Az timeseries 1... ',end='',flush=True)
    az1_data, az1_atr = readfile.read(args.az_ts1_file, datasetName='timeseries')
    az1_data -= az1_data[0]
    az1_data *= float(az1_atr['X_STEP'])
    print('done ')

    if len(args.rg_rs1_file) > 0 and len(args.az_rs1_file) > 0:
        print('Reading Rg residuals 1... ',end='',flush=True)
        rg1_res_data, rg1_res_atr = readfile.read(args.rg_rs1_file, datasetName='residual')
        rg1_res_data *= float(rg1_res_atr['X_STEP'])
        print('done ')

        print('Reading Az residuals 1... ',end='',flush=True)
        az1_res_data, az1_res_atr = readfile.read(args.az_rs1_file, datasetName='residual')
        az1_res_data *= float(az1_res_atr['X_STEP'])
        print('done ')

    print('Reading Rg timeseries 2... ',end='',flush=True)
    rg2_data, rg2_atr = readfile.read(args.rg_ts2_file, datasetName='timeseries')
    rg2_data -= rg2_data[0] #rg_data[0] should be all 0
    rg2_data *= float(rg2_atr['X_STEP'])
    print('done ')

    print('Reading Az timeseries 2... ',end='',flush=True)
    az2_data, az2_atr = readfile.read(args.az_ts2_file, datasetName='timeseries')
    az2_data -= az2_data[0]
    az2_data *= float(az2_atr['X_STEP'])
    print('done ')

    if len(args.rg_rs2_file) > 0 and len(args.az_rs2_file) > 0:
        print('Reading Rg residuals 2... ',end='',flush=True)
        rg2_res_data, rg2_res_atr = readfile.read(args.rg_rs2_file, datasetName='residual')
        rg2_res_data *= float(rg2_res_atr['X_STEP'])
        print('done ')

        print('Reading Az residuals 2... ',end='',flush=True)
        az2_res_data, az2_res_atr = readfile.read(args.az_rs2_file, datasetName='residual')
        az2_res_data *= float(az2_res_atr['X_STEP'])
        print('done ')


    print('Reading mask... ',end='',flush=True)
    if args.mask_file.endswith('.h5'):
        mask_data, mask_atr = readfile.read(args.mask_file, datasetName='mask')
    elif args.mask_file.endswith('.npy'):
        f = args.mask_file
        mask_data = np.load(f)
        f = None
    elif args.mask_file.endswith('.npy.gz'):
        f = gzip.GzipFile(args.mask_file, "r")
        mask_data = np.load(f)
        f = None
    elif args.mask_file.endswith('.tif'):
        from osgeo import gdal
        ds = gdal.Open(args.mask_file)
        mask_data = ds.GetRasterBand(1).ReadAsArray().shape
        ds = None

    if mask_data.shape == rg1_data[0].shape == False:
        print('Mask band and time series 1 need to have the same dimension.')
    if mask_data.shape == rg2_data[0].shape == False:
        print('Mask band and time series 2 need to have the same dimension.')
    print('done ')

    print('Masking %02d TS1 arrays... '%az1_data.shape[0])
    nre1 = mask_data[mask_data == 1].shape[0]
    az1_data_cum = np.empty((az1_data.shape[0], nre1), dtype=np.float32)
    az1_data_cum.fill(np.nan)
    rg1_data_cum = np.empty((rg1_data.shape[0], nre1), dtype=np.float32)
    rg1_data_cum.fill(np.nan)
    ts1_delta_year_ar = np.empty(az1_data.shape[0], dtype=np.float32)
    ts1_delta_year_ar.fill(np.nan)
    ts1_delta_day_ar = np.empty(az1_data.shape[0], dtype=np.float32)
    ts1_delta_day_ar.fill(np.nan)
    if len(args.rg_rs1_file) > 0 and len(args.az_rs1_file) > 0:
        az1_res = np.empty(nre1, dtype=np.float32)
        az1_res.fill(np.nan)
        rg1_res = np.empty(nre1, dtype=np.float32)
        rg1_res.fill(np.nan)
    for i in tqdm.tqdm(range(az1_data.shape[0])):
        if i == 0:
            caz_data = np.zeros(nre1)
            crg_data = np.zeros(nre1)
            caz_data_cum = np.zeros(nre1)
            crg_data_cum = np.zeros(nre1)
            delta_year = 0
            delta_day = 0
            if len(args.rg_rs1_file) > 0 and len(args.az_rs1_file) > 0:
                rg1_res = rg1_res_data[mask_data == 1]
                az1_res = az1_res_data[mask_data == 1]
        elif i > 0:
            delta_days = datetime.datetime.strptime(dates1[i], "%Y%m%d") - datetime.datetime.strptime(dates1[i-1], "%Y%m%d")
            delta_year = delta_days.days/365
            delta_day = delta_days.days
            # Load cumulative offset and mask
            caz_data_cum = az1_data[i,:,:]
            caz_data_cum = caz_data_cum[mask_data == 1]
            crg_data_cum = rg1_data[i,:,:]
            crg_data_cum = crg_data_cum[mask_data == 1]
            # could convert offsets to velocity - but we do linear interpolation for smoothing
            # caz_data = (az_data[i,:,:] - az_data[i-1,:,:]) / delta_year
            # caz_data = caz_data[mask_data == 1]
            # crg_data = (rg_data[i,:,:] - rg_data[i-1,:,:]) / delta_year
            # crg_data = crg_data[mask_data == 1]
        ts1_delta_year_ar[i] = delta_year
        ts1_delta_day_ar[i] = delta_day
        rg1_data_cum[i,:] = crg_data_cum
        az1_data_cum[i,:] = caz_data_cum

    #use residual filter and remove all points that are above residual percentage
    #maybe better to work with sum of residuals - here we are filtering separtely:
    # first for range and then for azimuth
    if len(args.rg_rs1_file) > 0 and len(args.az_rs1_file) > 0:
        #residual files are given - use these to apply additional filters for mask region
        residual_threshold = args.prcp_threshold
        rg1_res_p = np.percentile(rg1_res, residual_threshold)
        rg1_res_p_idx = np.where(rg1_res < rg1_res_p)[0]
        rg1_data_cum = rg1_data_cum[:,rg1_res_p_idx]
        az1_data_cum = az1_data_cum[:,rg1_res_p_idx]
        az1_res = az1_res[rg1_res_p_idx]
        #now azimuth
        az1_res_p = np.percentile(az1_res, residual_threshold)
        az1_res_p_idx = np.where(az1_res < az1_res_p)[0]
        rg1_data_cum = rg1_data_cum[:,az1_res_p_idx]
        az1_data_cum = az1_data_cum[:,az1_res_p_idx]
        az1_res = az1_res[az1_res_p_idx]

        #recalculate nre, because values have been removed
        nre1 = rg1_data_cum.shape[1]

    print('Masking %02d TS2 arrays... '%az2_data.shape[0])
    nre2 = mask_data[mask_data == 1].shape[0]
    az2_data_cum = np.empty((az2_data.shape[0], nre2), dtype=np.float32)
    az2_data_cum.fill(np.nan)
    rg2_data_cum = np.empty((rg2_data.shape[0], nre2), dtype=np.float32)
    rg2_data_cum.fill(np.nan)
    ts2_delta_year_ar = np.empty(az2_data.shape[0], dtype=np.float32)
    ts2_delta_year_ar.fill(np.nan)
    ts2_delta_day_ar = np.empty(az2_data.shape[0], dtype=np.float32)
    ts2_delta_day_ar.fill(np.nan)
    if len(args.rg_rs2_file) > 0 and len(args.az_rs2_file) > 0:
        az2_res = np.empty(nre2, dtype=np.float32)
        az2_res.fill(np.nan)
        rg2_res = np.empty(nre2, dtype=np.float32)
        rg2_res.fill(np.nan)
    for i in tqdm.tqdm(range(az2_data.shape[0])):
        if i == 0:
            caz_data = np.zeros(nre2)
            crg_data = np.zeros(nre2)
            caz_data_cum = np.zeros(nre2)
            crg_data_cum = np.zeros(nre2)
            delta_year = 0
            delta_day = 0
            if len(args.rg_rs2_file) > 0 and len(args.az_rs2_file) > 0:
                rg2_res = rg2_res_data[mask_data == 1]
                az2_res = az2_res_data[mask_data == 1]
        elif i > 0:
            delta_days = datetime.datetime.strptime(dates1[i], "%Y%m%d") - datetime.datetime.strptime(dates1[i-1], "%Y%m%d")
            delta_year = delta_days.days/365
            delta_day = delta_days.days
            # Load cumulative offset and mask
            caz_data_cum = az2_data[i,:,:]
            caz_data_cum = caz_data_cum[mask_data == 1]
            crg_data_cum = rg2_data[i,:,:]
            crg_data_cum = crg_data_cum[mask_data == 1]
            # could convert offsets to velocity - but we do linear interpolation for smoothing
            # caz_data = (az_data[i,:,:] - az_data[i-1,:,:]) / delta_year
            # caz_data = caz_data[mask_data == 1]
            # crg_data = (rg_data[i,:,:] - rg_data[i-1,:,:]) / delta_year
            # crg_data = crg_data[mask_data == 1]
        ts2_delta_year_ar[i] = delta_year
        ts2_delta_day_ar[i] = delta_day
        rg2_data_cum[i,:] = crg_data_cum
        az2_data_cum[i,:] = caz_data_cum

    if len(args.rg_rs2_file) > 0 and len(args.az_rs2_file) > 0:
        #residual files are given - use these to apply additional filters for mask region
        residual_threshold = args.prcp_threshold
        rg2_res_p = np.nanpercentile(rg2_res, residual_threshold)
        rg2_res_p_idx = np.where(rg2_res < rg2_res_p)[0]
        rg2_data_cum = rg2_data_cum[:,rg2_res_p_idx]
        az2_data_cum = az2_data_cum[:,rg2_res_p_idx]
        az2_res = az2_res[rg2_res_p_idx]
        #now azimuth
        az2_res_p = np.percentile(az2_res, residual_threshold)
        az2_res_p_idx = np.where(az2_res < az2_res_p)[0]
        rg2_data_cum = rg2_data_cum[:,az2_res_p_idx]
        az2_data_cum = az2_data_cum[:,az2_res_p_idx]
        az2_res = az2_res[az2_res_p_idx]

        #recalculate nre2, because values have been removed
        nre2 = rg2_data_cum.shape[1]


    # perform linear interpolation of dx and dy offsets using delta_years
    tsamples1_monthly = np.arange(0, np.cumsum(ts1_delta_day_ar).max()+1, 30) # for interpolation
    #linear interpolation for each pixel from mask:
    az1_data_cum_li, rg1_data_cum_li = dxdy_linear_interpolation(tsamples1_monthly, ts1_delta_day_ar, rg1_data_cum, az1_data_cum, nre1)

    #make sure that beginning dates match (dates1[0] and dates2[0] will need to be the same)
    tsamples2_monthly = np.arange(0, np.cumsum(ts2_delta_day_ar).max()+1, 30) # for interpolation
    #linear interpolation for each pixel from mask:
    az2_data_cum_li, rg2_data_cum_li = dxdy_linear_interpolation(tsamples2_monthly, ts2_delta_day_ar, rg2_data_cum, az2_data_cum, nre2)

    # # For Debugging: plot mean time series
    fig, ax = plt.subplots(1,2, figsize = (12, 8), dpi=300)
    ax[0].plot(np.cumsum(ts1_delta_day_ar), np.mean(rg1_data_cum, axis=1), 'x', color='darkblue', label='ts1 rg data (var weights)')
    ax[0].plot(np.cumsum(ts2_delta_day_ar), np.mean(rg2_data_cum, axis=1), 'x', color='darkred', label='ts2 rg data (no weights)')
    ax[0].plot(tsamples1_monthly, np.mean(rg1_data_cum_li, axis=1), '-', color='darkblue', lw=2, label='ts1 monthly linear interpolation (var weights)')
    ax[0].plot(tsamples2_monthly, np.mean(rg2_data_cum_li, axis=1), '-', color='darkred', lw=2, label='ts2 monthly linear interpolation (no weights)')
    ax[0].grid()
    ax[0].legend()
    ax[0].set_ylabel("Cumulative displacement dx [pix]")
    ax[0].set_xlabel("Time [days]")
    ax[1].plot(np.cumsum(ts1_delta_day_ar), np.mean(az1_data_cum, axis=1), 'x', color='darkblue', label='ts1 az data (var weights)')
    ax[1].plot(np.cumsum(ts1_delta_day_ar), np.mean(az2_data_cum, axis=1), 'x', color='darkred', label='ts2 az data (no weights)')
    ax[1].plot(tsamples1_monthly, np.mean(az1_data_cum_li, axis=1), '-', color='darkblue', lw=2, label='ts1 monthly linear interpolation (var weights)')
    ax[1].plot(tsamples2_monthly, np.mean(az2_data_cum_li, axis=1), '-', color='darkred', lw=2, label='ts2 monthly linear interpolation (no weights)')
    ax[1].set_ylabel("Cumulative displacement dy [pix]")
    ax[1].set_xlabel("Time [days]")
    ax[1].grid()
    ax[1].legend()
    fig.suptitle('Cumulative TS from inversion with and without weights', fontsize=16)
    fig.tight_layout()
    fig.savefig(args.out1_pngfname, dpi=300)

    print('calculate velocity from linearily interpolated offset values: TS1')
    # use interpolated values to calculate velocity and direction
    v_mag1 = np.empty((tsamples1_monthly.shape[0], nre1), dtype=np.float32)
    v_mag1.fill(np.nan)
    v_dir1 = np.empty((tsamples1_monthly.shape[0], nre1), dtype=np.float32)
    v_dir1.fill(np.nan)
    for i in tqdm.tqdm(range(tsamples1_monthly.shape[0])):
        if i == 0:
            cv_mag = np.zeros(nre1)
            cv_dir = np.zeros(nre1)
        elif i > 0:
            delta_day = tsamples1_monthly[i] - tsamples1_monthly[i-1]
            delta_year = delta_days.days/365
            caz1_data_cum_li = (az1_data_cum_li[i,:] - az1_data_cum_li[i-1,:]) / delta_year
            crg1_data_cum_li = (rg1_data_cum_li[i,:] - rg1_data_cum_li[i-1,:]) / delta_year
            v_mag1[i,:] = np.sqrt(caz1_data_cum_li**2 + crg1_data_cum_li**2)
            v_dir1[i,:] = np.rad2deg(np.arctan2(caz1_data_cum_li, crg1_data_cum_li))

    #convert number of days (from interpolation) to datetime
    base1 = pd.to_datetime(dates1[0], format='%Y%m%d')
    date1_list = []
    date1_list.append(base1)
    for i in range(1,tsamples1_monthly.shape[0]):
        date1_list.append(base1 + datetime.timedelta(days=tsamples1_monthly[i]))
    vdata1 = np.c_[tsamples1_monthly, np.nanmean(az1_data_cum_li, axis=1), np.nanstd(az1_data_cum_li, axis=1),
            np.nanmean(rg1_data_cum_li, axis=1), np.nanstd(rg1_data_cum_li, axis=1),
            np.nanmean(v_mag1, axis=1), np.nanstd(v_mag1, axis=1),
            np.nanpercentile(v_mag1, axis=1, q=25), np.nanpercentile(v_mag1, axis=1, q=75),
            np.nanmean(v_dir1, axis=1), np.nanstd(v_dir1, axis=1)]
    vel_mag_dir_df1 = pd.DataFrame(vdata1, columns=['nr_of_days', 'dy_mean', 'dy_std', 'dx_mean', 'dx_std', 'v_mean', 'v_std', 'v_25p', 'v_75p', 'dir_mean', 'dir_std'],
                                            index=date1_list)

    print('calculate velocity from linearily interpolated offset values: TS2')
    # use interpolated values to calculate velocity and direction
    v_mag2 = np.empty((tsamples2_monthly.shape[0], nre2), dtype=np.float32)
    v_mag1.fill(np.nan)
    v_dir2 = np.empty((tsamples2_monthly.shape[0], nre2), dtype=np.float32)
    v_dir2.fill(np.nan)
    for i in tqdm.tqdm(range(tsamples2_monthly.shape[0])):
        if i == 0:
            cv_mag = np.zeros(nre2)
            cv_dir = np.zeros(nre2)
        elif i > 0:
            delta_day = tsamples2_monthly[i] - tsamples2_monthly[i-1]
            delta_year = delta_days.days/365
            caz2_data_cum_li = (az2_data_cum_li[i,:] - az2_data_cum_li[i-1,:]) / delta_year
            crg2_data_cum_li = (rg2_data_cum_li[i,:] - rg2_data_cum_li[i-1,:]) / delta_year
            v_mag2[i,:] = np.sqrt(caz2_data_cum_li**2 + crg2_data_cum_li**2)
            v_dir2[i,:] = np.rad2deg(np.arctan2(caz2_data_cum_li, crg2_data_cum_li))

    #convert number of days (from interpolation) to datetime
    base2 = pd.to_datetime(dates2[0], format='%Y%m%d')
    date2_list = []
    date2_list.append(base2)
    for i in range(1,tsamples2_monthly.shape[0]):
        date2_list.append(base2 + datetime.timedelta(days=tsamples2_monthly[i]))
    vdata2 = np.c_[tsamples2_monthly, np.nanmean(az2_data_cum_li, axis=1), np.nanstd(az2_data_cum_li, axis=1),
            np.nanmean(rg2_data_cum_li, axis=1), np.nanstd(rg2_data_cum_li, axis=1),
            np.nanmean(v_mag2, axis=1), np.nanstd(v_mag2, axis=1),
            np.nanpercentile(v_mag2, axis=1, q=25), np.nanpercentile(v_mag2, axis=1, q=75),
            np.nanmean(v_dir2, axis=1), np.nanstd(v_dir2, axis=1)]
    vel_mag_dir_df2 = pd.DataFrame(vdata2, columns=['nr_of_days', 'dy_mean', 'dy_std', 'dx_mean', 'dx_std', 'v_mean', 'v_std', 'v_25p', 'v_75p', 'dir_mean', 'dir_std'],
                                            index=date2_list)

    fig, ax = plt.subplots(1, 3, figsize = (16, 8), dpi=300)
    # ax[0].errorbar(vel_mag_dir_df1.index, vel_mag_dir_df1.dx_mean, yerr=vel_mag_dir_df1.dx_std, color='darkblue', lw=0.5)
    ax[0].plot(vel_mag_dir_df1.index, vel_mag_dir_df1.dx_mean, '-', lw=2, color='lightcoral')
    ax[0].plot(pd.to_datetime(dates1, format='%Y%m%d'), np.nanmean(rg1_data_cum, axis=1), 'x', color='darkred', label='raw rg1 data')
    ax[0].plot(vel_mag_dir_df2.index, vel_mag_dir_df2.dx_mean, '-', lw=2, color='lightblue')
    ax[0].plot(pd.to_datetime(dates2, format='%Y%m%d'), np.nanmean(rg2_data_cum, axis=1), 'x', color='darkblue', label='raw rg2 data')
    ax[0].grid()
    ax[0].legend()
    ax[0].set_ylabel("Cumulative displacement dx [pix]")
    ax[0].set_xlabel("Time")
    # ax[1].errorbar(vel_mag_dir_df1.index, vel_mag_dir_df1.dy_mean, yerr=vel_mag_dir_df1.dy_std, color='darkblue', lw=0.5)
    ax[1].plot(vel_mag_dir_df1.index, vel_mag_dir_df1.dy_mean, '-', lw=2, color='lightcoral')
    ax[1].plot(pd.to_datetime(dates1, format='%Y%m%d'), np.nanmean(az1_data_cum, axis=1), 'x', color='darkred', label='raw az1 data')
    ax[1].plot(vel_mag_dir_df2.index, vel_mag_dir_df2.dy_mean, '-', lw=2, color='lightblue')
    ax[1].plot(pd.to_datetime(dates2, format='%Y%m%d'), np.nanmean(az2_data_cum, axis=1), 'x', color='darkblue', label='raw az2 data')
    ax[1].set_ylabel("Cumulative displacement dy [pix]")
    ax[1].grid()
    ax[1].legend()
    ax[1].set_xlabel("Time")
    # ax[2].errorbar(vel_mag_dir_df1.index, vel_mag_dir_df1.v_mean, yerr=vel_mag_dir_df1.v_std, color='darkblue', lw=0.5)
    ax[2].plot(vel_mag_dir_df1.index[1:], vel_mag_dir_df1.v_mean[1:], 'darkred', lw=2, label='TS1 Velocity')
    ax[2].plot(vel_mag_dir_df2.index[1:], vel_mag_dir_df2.v_mean[1:], 'darkblue', lw=2, label='TS1 Velocity')
    ax[2].set_ylabel("Velocity [m/y]")
    ax[2].grid()
    ax[2].set_xlabel("Time")
    fig.suptitle('Cumulative TS with linear interpolation from inversion', fontsize=16)
    fig.tight_layout()
    fig.savefig(args.out2_pngfname)


    # # prepare pcolormesh with histogram plot for velocity
    # mintimedelta = 30 # [days]
    # nbins = 100
    # v = v_mag[1:,:]
    # b = np.linspace(v.min(), v.max(), nbins+1)
    #
    # #d = [datetime.datetime.strptime(s, '%Y%m%d') for s in dates]
    # d = date_list
    # d = d[1:]
    # ndays = (d[-1]-d[0]).days+1
    # x = [d[0] + datetime.timedelta(days = i*mintimedelta) for i in range(ndays//mintimedelta+2)]
    # img = np.nan * np.ones((nbins, ndays//mintimedelta+1))
    # for j, di in enumerate(d):
    #     i = (di-d[0]).days//mintimedelta
    #     img[:, i], _ = np.histogram(v[j,:], b, density = True)
    #
    # dtime_idx = np.empty(len(dates[1::]), dtype=np.int16)
    # for i in range(len(dates[1::])):
    #     dtime_idx[i] = np.argmin(np.abs(pd.to_datetime(x)-pd.to_datetime(dates[i+1])))
    #
    #
    # fig, ax = plt.subplots(1, 3, figsize = (19.2, 10.8), dpi=300)
    # im0 = ax[0].pcolormesh(x, b, img, norm = LogNorm(), cmap = plt.cm.magma_r)
    # ax[0].plot(vel_mag_dir_df.index, vel_mag_dir_df.v_mean, 'w', lw=2)
    # ax[0].plot(vel_mag_dir_df.index[dtime_idx], vel_mag_dir_df.v_mean[dtime_idx], 'wx', ms=5, lw=1)
    # ax[0].xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=12))
    # ax[0].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
    # ax[0].xaxis.set_minor_locator(matplotlib.dates.MonthLocator(interval=1))
    # # ax[0].xaxis.set_minor_formatter(matplotlib.dates.DateFormatter('%m'))
    # cb0 = plt.colorbar(im0, ax=ax[0], location='bottom', pad=0.1)
    # cb0.set_label('Probability')
    # ax[0].set_title('Lin. Interp. Velocity from inversion with var weights')
    # ax[0].set_xlabel('Time')
    # ax[0].set_ylabel('Velocity [m/y]')
    # ax[0].grid()
    #
    # # prepare pcolormesh with histogram plot for dx cumulative
    # v2 = rg_data_cum_li[1:,:]
    # b = np.linspace(v2.min(), v2.max(), nbins+1)
    # img2 = np.nan * np.ones((nbins, ndays//mintimedelta+1))
    # for j, di in enumerate(d):
    #     i = (di-d[0]).days//mintimedelta
    #     img2[:, i], _ = np.histogram(v2[j,:], b, density = True)
    #
    # im1 = ax[1].pcolormesh(x, b, img2, norm = LogNorm(), cmap = plt.cm.viridis)
    # ax[1].plot(vel_mag_dir_df.index, vel_mag_dir_df.dx_mean, 'k', lw=2)
    # ax[1].plot(vel_mag_dir_df.index[dtime_idx], vel_mag_dir_df.dx_mean[dtime_idx], 'wx', ms=5, lw=1)
    # ax[1].xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=12))
    # ax[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
    # ax[1].xaxis.set_minor_locator(matplotlib.dates.MonthLocator(interval=1))
    # # ax[1].xaxis.set_minor_formatter(matplotlib.dates.DateFormatter('%m'))
    # cb1 = plt.colorbar(im1, ax=ax[1], location='bottom', pad=0.1)
    # cb1.set_label('Probability')
    # ax[1].set_title('Lin. Interp. cumulative dx from inversion')
    # ax[1].set_xlabel('Time')
    # ax[1].set_ylabel('dx [m]')
    # ax[1].grid()
    #
    # # prepare pcolormesh with histogram plot for dy cumulative
    # v3 = az_data_cum_li[1:,:]
    # b = np.linspace(v3.min(), v3.max(), nbins+1)
    # img3 = np.nan * np.ones((nbins, ndays//mintimedelta+1))
    # for j, di in enumerate(d):
    #     i = (di-d[0]).days//mintimedelta
    #     img3[:, i], _ = np.histogram(v3[j,:], b, density = True)
    #
    # im2 = ax[2].pcolormesh(x, b, img3, norm = LogNorm(), cmap = plt.cm.viridis)
    # ax[2].plot(vel_mag_dir_df.index, vel_mag_dir_df.dy_mean, 'k', lw=2)
    # ax[2].plot(vel_mag_dir_df.index[dtime_idx], vel_mag_dir_df.dy_mean[dtime_idx], 'wx', ms=5, lw=1)
    # ax[2].xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=12))
    # ax[2].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
    # ax[2].xaxis.set_minor_locator(matplotlib.dates.MonthLocator(interval=1))
    # # ax[2].xaxis.set_minor_formatter(matplotlib.dates.DateFormatter('%m'))
    # cb2 = plt.colorbar(im2, ax=ax[2], location='bottom', pad=0.1)
    # cb2.set_label('Probability')
    # ax[2].set_title('Lin. Interp. cumulative dy from inversion')
    # ax[2].set_xlabel('Time')
    # ax[2].set_ylabel('dy [m]')
    # ax[2].grid()
    # fig.tight_layout()
    # fig.savefig(args.out_pngfname, dpi=300)
