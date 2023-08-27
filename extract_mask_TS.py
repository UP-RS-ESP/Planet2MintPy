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
extract_mask_TS.py \
    --az_ts_file timeseriesAz_var.h5 \
    --rg_ts_file timeseriesRg_var.h5 \
    --mask_file aoi4_var_velocity_mask.h5 \
    --HDF_outfile  aoi4_var_velocity_mask_ts.h5 \
    --npy_outfile aoi4_var_velocity_mask_ts.npy \
    --out_pngfname aoi4_var_velocity_mask_ts.png
"""

DESCRIPTION = """
Extract time series for masked region from dx (range) and dy (azimuth). Mask dataset need to be same size as TS.
Save clipped dataset to npy file and create plot.

Aug-2023, Bodo Bookhagen (bodo.bookhagen@uni-potsdam.de) and Ariane Mueting (mueting@uni-potsdam.de)
"""

def cmdLineParser():
    from argparse import RawTextHelpFormatter
    parser = argparse.ArgumentParser(description=DESCRIPTION, epilog=EXAMPLE, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--rg_ts_file', help='dx (Rg) offset timeseries', required=True)
    parser.add_argument('--az_ts_file', help='dy (Az) offset timeseries', required=True)
    parser.add_argument('--mask_file', help='Mask file with data == 1', required=True)
    parser.add_argument('--HDF_outfile', help='Output filename containing percentiles of each timestep', required=True)
    parser.add_argument('--npy_outfile', help='Output filename for masked array from each timestep', required=True)
    parser.add_argument('--out_pngfname', default="", help='Output TS plot in PNG format', required=True)
    return parser.parse_args()


@njit(parallel=True)
def dxdy_linear_interpolation(tsamples_monthly, delta_day_ar, rg_data_cum, az_data_cum, nre):
    az_data_cum_li = np.empty((tsamples_monthly.shape[0], nre), dtype=np.float32)
    az_data_cum_li.fill(np.nan)
    rg_data_cum_li = np.empty((tsamples_monthly.shape[0], nre), dtype=np.float32)
    rg_data_cum_li.fill(np.nan)
    for i in prange(rg_data_cum.shape[1]):
        #iterate through each pixel in mask and perform linear interpolation for dx and dy
        fit_dx = np.interp(x=tsamples_monthly, xp=np.cumsum(delta_day_ar), fp=rg_data_cum[:,i]) #using numpy - it's faster with numba
        fit_dy = np.interp(x=tsamples_monthly, xp=np.cumsum(delta_day_ar), fp=az_data_cum[:,i]) #using numpy - it's faster with numba
        az_data_cum_li[:,i] =  fit_dy
        rg_data_cum_li[:,i] =  fit_dx
    return az_data_cum_li, rg_data_cum_li


if __name__ == '__main__':
    np.seterr(divide='ignore', invalid='ignore')
    warnings.simplefilter("ignore")

    args = cmdLineParser()

    # #testing purposes:
    # parser = argparse.ArgumentParser(description='Extract time series from mask (mask needs to be same size as TS).')
    # args = parser.parse_args()
    # #args.az_ts_file = '/raid/L8_DelMedio/mintpy/timeseriesAz_var.h5'
    # #args.rg_ts_file = '/raid/L8_DelMedio/mintpy/timeseriesRg_var.h5'
    # #args.mask_file = '/raid/L8_DelMedio/mintpy/DelMedio_var_velocity_mask.h5'
    # #args.HDF_outfile = '/raid/L8_DelMedio/mintpy/DelMedio_var_velocity_masked_ts.h5'
    # args.az_ts_file = 'timeseriesAz_var.h5'
    # args.rg_ts_file = 'timeseriesRg_var.h5'
    # args.mask_file = 'aoi4_var_velocity_mask.h5'
    # args.HDF_outfile = 'aoi4_var_velocity_mask_ts.h5'
    # args.out_pngfname='aoi4_var_velocity_mask_ts_velocity.png'

    args.atr = readfile.read_attribute(args.az_ts_file)
    args.coord = ut.coordinate(args.atr)

    # input file info
    ts = ut.timeseries(file=args.az_ts_file)
    dates = ts.get_date_list()
    ts=None

    # Both time series contain cumulative offset. Will need to convert to velocity before analysis
    print('Reading Rg timeseries... ',end='',flush=True)
    rg_data, rg_atr = readfile.read(args.rg_ts_file, datasetName='timeseries')
    rg_data -= rg_data[0] #rg_data[0] should be all 0
    rg_data *= float(rg_atr['X_STEP'])
    print('done ')

    print('Reading Az timeseries... ',end='',flush=True)
    az_data, az_atr = readfile.read(args.az_ts_file, datasetName='timeseries')
    az_data -= az_data[0]
    az_data *= float(az_atr['X_STEP'])
    print('done ')

    print('Reading mask... ',end='',flush=True)
    mask_data, mask_atr = readfile.read(args.mask_file, datasetName='mask')
    if mask_data.shape == rg_data[0].shape == False:
        print('Mask band and time series need to have the same dimension.')
    print('done ')

    print('Masking %02d TS arrays... '%az_data.shape[0])
    nre = mask_data[mask_data == 1].shape[0]
    az_data_cum = np.empty((az_data.shape[0], nre), dtype=np.float32)
    az_data_cum.fill(np.nan)
    rg_data_cum = np.empty((rg_data.shape[0], nre), dtype=np.float32)
    rg_data_cum.fill(np.nan)
    delta_year_ar = np.empty(az_data.shape[0], dtype=np.float32)
    delta_year_ar.fill(np.nan)
    delta_day_ar = np.empty(az_data.shape[0], dtype=np.float32)
    delta_day_ar.fill(np.nan)
    for i in tqdm.tqdm(range(az_data.shape[0])):
        if i == 0:
            caz_data = np.zeros(nre)
            crg_data = np.zeros(nre)
            caz_data_cum = np.zeros(nre)
            crg_data_cum = np.zeros(nre)
            delta_year = 0
            delta_day = 0
        elif i > 0:
            delta_days = datetime.datetime.strptime(dates[i], "%Y%m%d") - datetime.datetime.strptime(dates[i-1], "%Y%m%d")
            delta_year = delta_days.days/365
            delta_day = delta_days.days
            # Load cumulative offset and mask
            caz_data_cum = az_data[i,:,:]
            caz_data_cum = caz_data_cum[mask_data == 1]
            crg_data_cum = rg_data[i,:,:]
            crg_data_cum = crg_data_cum[mask_data == 1]
            # could convert offsets to velocity - but we do linear interpolation for smoothing
            # caz_data = (az_data[i,:,:] - az_data[i-1,:,:]) / delta_year
            # caz_data = caz_data[mask_data == 1]
            # crg_data = (rg_data[i,:,:] - rg_data[i-1,:,:]) / delta_year
            # crg_data = crg_data[mask_data == 1]
        delta_year_ar[i] = delta_year
        delta_day_ar[i] = delta_day
        rg_data_cum[i,:] = crg_data_cum
        az_data_cum[i,:] = caz_data_cum

    # plt.plot(np.cumsum(delta_day_ar), np.mean(rg_data_cum, axis=1), 'k.'))
    # perform linear interpolation of dx and dy offsets using delta_years
    tsamples_monthly = np.arange(0, np.cumsum(delta_day_ar).max()+1, 30) # for interpolation
    #linear interpolation for each pixel from mask:
    az_data_cum_li, rg_data_cum_li = dxdy_linear_interpolation(tsamples_monthly, delta_day_ar, rg_data_cum, az_data_cum, nre)

    # For Debugging: plot mean time series
    # fig, ax = plt.subplots(1,2, figsize = (12, 8), dpi=300)
    # ax[0].plot(np.cumsum(delta_day_ar), np.mean(rg_data_cum, axis=1), 'b+', label='raw data')
    # ax[0].plot(tsamples_monthly, np.mean(rg_data_cum_li, axis=1), 'k', lw=2, label='monthly linear interpolation')
    # ax[0].grid()
    # ax[0].legend()
    # ax[0].set_ylabel("Cumulative displacement dx [pix]")
    # ax[0].set_xlabel("Time [days]")
    # ax[1].plot(np.cumsum(delta_day_ar), np.mean(az_data_cum, axis=1), 'b+', label='raw data')
    # ax[1].plot(tsamples_monthly, np.mean(az_data_cum_li, axis=1), 'k', lw=2, label='monthly linear interpolation')
    # ax[1].set_ylabel("Cumulative displacement dy [pix]")
    # ax[1].set_xlabel("Time [days]")
    # ax[1].grid()
    # ax[1].legend()
    # fig.suptitle('AOI4: Cumulative TS and linear monthly interpolation from inversion with weights', fontsize=16)
    # fig.tight_layout()

    # use interpolated values to calculate velocity and direction
    v_mag = np.empty((tsamples_monthly.shape[0], nre), dtype=np.float32)
    v_mag.fill(np.nan)
    v_dir = np.empty((tsamples_monthly.shape[0], nre), dtype=np.float32)
    v_dir.fill(np.nan)
    for i in tqdm.tqdm(range(tsamples_monthly.shape[0])):
        if i == 0:
            cv_mag = np.zeros(nre)
            cv_dir = np.zeros(nre)
        elif i > 0:
            delta_day = tsamples_monthly[i] - tsamples_monthly[i-1]
            delta_year = delta_days.days/365
            caz_data_cum_li = (az_data_cum_li[i,:] - az_data_cum_li[i-1,:]) / delta_year
            crg_data_cum_li = (rg_data_cum_li[i,:] - rg_data_cum_li[i-1,:]) / delta_year
            v_mag[i,:] = np.sqrt(caz_data_cum_li**2 + crg_data_cum_li**2)
            v_dir[i,:] = np.rad2deg(np.arctan2(caz_data_cum_li, crg_data_cum_li))

    #save matrices to numpy arrays
    v_mag_fname = args.npy_outfile[:-4] + '_vmag.npy.gz'
    f = gzip.GzipFile(v_mag_fname, "w")
    np.save(file=f, arr=v_mag)
    f.close()
    f = None
    v_dir_fname = args.npy_outfile[:-4] + '_vdir.npy.gz'
    f = gzip.GzipFile(v_dir_fname, "w")
    np.save(file=f, arr=v_dir)
    f.close()
    f = None
    az_fname = args.npy_outfile[:-4] + '_az_li.npy.gz'
    f = gzip.GzipFile(az_fname, "w")
    np.save(file=f, arr=az_data_cum_li)
    f.close()
    f = None
    rg_fname = args.npy_outfile[:-4] + '_rg_li.npy.gz'
    f = gzip.GzipFile(rg_fname, "w")
    np.save(file=f, arr=rg_data_cum_li)
    f.close()
    f = None

    # Can be loaded with:
    # f = gzip.GzipFile(date0_stack_fname, "r")
    # date0_stack = np.load(f)
    # f = None

    #convert number of days (from interpolation) to datetime
    base = pd.to_datetime(dates[0], format='%Y%m%d')
    date_list = []
    date_list.append(base)
    for i in range(1,tsamples_monthly.shape[0]):
        date_list.append(base + datetime.timedelta(days=tsamples_monthly[i]))
    # store mean and statistics into pandas HDF file
    #not storing original value, but only interpolated values
    # vdata = np.c_[np.nanmean(tsamples_monthly, az_data_cum, axis=1), np.nanstd(az_data_cum, axis=1),
    #         np.nanmean(rg_data_cum, axis=1), np.nanstd(rg_data_cum, axis=1),
    #         np.nanmean(rg_data_cum_li, axis=1), np.nanstd(rg_data_cum_li, axis=1),
    #         np.nanmean(rg_data_cum, axis=1), np.nanstd(rg_data_cum, axis=1),
    #         np.nanmean(vel_magnitude, axis=1), np.nanstd(vel_magnitude, axis=1),
    #         np.nanpercentile(vel_magnitude, axis=1, q=25), np.nanpercentile(vel_magnitude, axis=1, q=75),
    #         np.nanmean(vel_direction, axis=1), np.nanstd(vel_direction, axis=1)]
    vdata = np.c_[tsamples_monthly, np.nanmean(az_data_cum_li, axis=1), np.nanstd(az_data_cum_li, axis=1),
            np.nanmean(rg_data_cum_li, axis=1), np.nanstd(rg_data_cum_li, axis=1),
            np.nanmean(v_mag, axis=1), np.nanstd(v_mag, axis=1),
            np.nanpercentile(v_mag, axis=1, q=25), np.nanpercentile(v_mag, axis=1, q=75),
            np.nanmean(v_dir, axis=1), np.nanstd(v_dir, axis=1)]
    vel_mag_dir_df = pd.DataFrame(vdata, columns=['nr_of_days', 'dy_mean', 'dy_std', 'dx_mean', 'dx_std', 'v_mean', 'v_std', 'v_25p', 'v_75p', 'dir_mean', 'dir_std'],
                                            index=date_list)
    vel_mag_dir_df.to_hdf(args.HDF_outfile, key='vel_mag_dir_df', complevel=7)

    fig, ax = plt.subplots(1, 3, figsize = (16, 8), dpi=300)
    ax[0].errorbar(vel_mag_dir_df.index, vel_mag_dir_df.dx_mean, yerr=vel_mag_dir_df.dx_std, color='navy', lw=0.5)
    ax[0].plot(vel_mag_dir_df.index, vel_mag_dir_df.dx_mean, 'k', lw=2)
    ax[0].grid()
    ax[0].set_ylabel("Cumulative displacement dx [pix]")
    ax[0].set_xlabel("Time")
    ax[1].errorbar(vel_mag_dir_df.index, vel_mag_dir_df.dy_mean, yerr=vel_mag_dir_df.dy_std, color='navy', lw=0.5)
    ax[1].plot(vel_mag_dir_df.index, vel_mag_dir_df.dy_mean, 'k', lw=2)
    ax[1].set_ylabel("Cumulative displacement dy [pix]")
    ax[1].grid()
    ax[1].set_xlabel("Time")
    ax[2].errorbar(vel_mag_dir_df.index, vel_mag_dir_df.v_mean, yerr=vel_mag_dir_df.v_std, color='navy', lw=0.5)
    ax[2].plot(vel_mag_dir_df.index, vel_mag_dir_df.v_mean, 'k', lw=2)
    ax[2].set_ylabel("Velocity [m/y]")
    ax[2].grid()
    ax[2].set_xlabel("Time")
    fig.suptitle('Cumulative TS with linear interpolation from inversion with weights', fontsize=16)
    fig.tight_layout()
    fig.savefig(args.out_pngfname[:-4]+'_mean.png')


    # prepare pcolormesh with histogram plot for velocity
    mintimedelta = 30 # [days]
    nbins = 100
    v = v_mag[1:,:]
    b = np.linspace(v.min(), v.max(), nbins+1)

    #d = [datetime.datetime.strptime(s, '%Y%m%d') for s in dates]
    d = date_list
    d = d[1:]
    ndays = (d[-1]-d[0]).days+1
    x = [d[0] + datetime.timedelta(days = i*mintimedelta) for i in range(ndays//mintimedelta+2)]
    img = np.nan * np.ones((nbins, ndays//mintimedelta+1))
    for j, di in enumerate(d):
        i = (di-d[0]).days//mintimedelta
        img[:, i], _ = np.histogram(v[j,:], b, density = True)

    fig, ax = plt.subplots(1, 3, figsize = (19.2, 10.8), dpi=300)
    im0 = ax[0].pcolormesh(x, b, img, norm = LogNorm(), cmap = plt.cm.magma_r)
    ax[0].plot(vel_mag_dir_df.index, vel_mag_dir_df.v_mean, 'w', lw=2)
    ax[0].xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=12))
    ax[0].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
    ax[0].xaxis.set_minor_locator(matplotlib.dates.MonthLocator(interval=1))
    # ax[0].xaxis.set_minor_formatter(matplotlib.dates.DateFormatter('%m'))
    cb0 = plt.colorbar(im0, ax=ax[0], location='bottom', pad=0.1)
    cb0.set_label('Probability')
    ax[0].set_title('Lin. Interp. Velocity from inversion with var weights')
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Velocity [m/y]')
    ax[0].grid()

    # prepare pcolormesh with histogram plot for dx cumulative
    v2 = rg_data_cum_li[1:,:]
    b = np.linspace(v2.min(), v2.max(), nbins+1)
    img2 = np.nan * np.ones((nbins, ndays//mintimedelta+1))
    for j, di in enumerate(d):
        i = (di-d[0]).days//mintimedelta
        img2[:, i], _ = np.histogram(v2[j,:], b, density = True)

    im1 = ax[1].pcolormesh(x, b, img2, norm = LogNorm(), cmap = plt.cm.viridis)
    ax[1].plot(vel_mag_dir_df.index, vel_mag_dir_df.dx_mean, 'k', lw=2)
    ax[1].xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=12))
    ax[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
    ax[1].xaxis.set_minor_locator(matplotlib.dates.MonthLocator(interval=1))
    # ax[1].xaxis.set_minor_formatter(matplotlib.dates.DateFormatter('%m'))
    cb1 = plt.colorbar(im1, ax=ax[1], location='bottom', pad=0.1)
    cb1.set_label('Probability')
    ax[1].set_title('Lin. Interp. cumulative dx from inversion')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('dx [m]')
    ax[1].grid()

    # prepare pcolormesh with histogram plot for dy cumulative
    v3 = az_data_cum_li[1:,:]
    b = np.linspace(v3.min(), v3.max(), nbins+1)
    img3 = np.nan * np.ones((nbins, ndays//mintimedelta+1))
    for j, di in enumerate(d):
        i = (di-d[0]).days//mintimedelta
        img3[:, i], _ = np.histogram(v3[j,:], b, density = True)

    im2 = ax[2].pcolormesh(x, b, img3, norm = LogNorm(), cmap = plt.cm.viridis)
    ax[2].plot(vel_mag_dir_df.index, vel_mag_dir_df.dy_mean, 'k', lw=2)
    ax[2].xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=12))
    ax[2].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
    ax[2].xaxis.set_minor_locator(matplotlib.dates.MonthLocator(interval=1))
    # ax[2].xaxis.set_minor_formatter(matplotlib.dates.DateFormatter('%m'))
    cb2 = plt.colorbar(im2, ax=ax[2], location='bottom', pad=0.1)
    cb2.set_label('Probability')
    ax[2].set_title('Lin. Interp. cumulative dy from inversion')
    ax[2].set_xlabel('Time')
    ax[2].set_ylabel('dy [m]')
    ax[2].grid()
    fig.tight_layout()
    fig.savefig(args.out_pngfname, dpi=300)
