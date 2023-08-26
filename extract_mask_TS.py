#!/usr/bin/env python
"""
@author: bodo bookhagen, bodo.bookhagen@uni-potsdam.de
"""
import warnings, argparse, os, tqdm, datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LogNorm
from mintpy.utils import arg_utils, readfile, utils as ut, plot as pp
from mintpy.defaults.plot import *
from scipy.interpolate import interp1d

import pandas as pd
import scipy.stats as stats
import matplotlib.dates as mdates
#import statsmodels.api as sm
#import seaborn as sns
from osgeo import gdal
#conda install -c conda-forge statsmodels
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

EXAMPLE = """example:
extract_mask_TS.py \
    --az_ts_file /raid/PS2_aoi3/mintpy/timeseriesAz_var.h5 \
    --rg_ts_file /raid/PS2_aoi3/mintpy/timeseriesRg_var.h5 \
    --mask_file /raid/PS2_aoi3/mintpy/aoi3_var_velocity_mask.h5 \
    --HDF_outfile  /raid/PS2_aoi6/mintpy/aoi3_var_velocity_mask_ts.h5 \
    --npy_outfile /raid/PS2_aoi6/mintpy/aoi3_var_velocity_mask_ts.npy \
    --out_pngfname /raid/PS2_aoi6/mintpy/aoi3_var_velocity_mask_ts.png
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

if __name__ == '__main__':
    np.seterr(divide='ignore', invalid='ignore')
    warnings.simplefilter("ignore")

    args = cmdLineParser()

    #testing purposes:
    parser = argparse.ArgumentParser(description='Extract time series from mask (mask needs to be same size as TS).')
    args = parser.parse_args()
    #args.az_ts_file = '/raid/L8_DelMedio/mintpy/timeseriesAz_var.h5'
    #args.rg_ts_file = '/raid/L8_DelMedio/mintpy/timeseriesRg_var.h5'
    #args.mask_file = '/raid/L8_DelMedio/mintpy/DelMedio_var_velocity_mask.h5'
    #args.HDF_outfile = '/raid/L8_DelMedio/mintpy/DelMedio_var_velocity_masked_ts.h5'
    args.az_ts_file = '/raid/PS2_aoi3/mintpy/timeseriesAz_var.h5'
    args.rg_ts_file = '/raid/PS2_aoi3/mintpy/timeseriesRg_var.h5'
    args.mask_file = '/raid/PS2_aoi3/mintpy/aoi3_var_velocity_mask.h5'
    args.HDF_outfile = '/raid/PS2_aoi6/mintpy/aoi3_var_velocity_mask_ts.h5'
    args.out_pngfname='aoi6_var_velocity_mask_ts_velocity.png'

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
    v_mag = np.empty((az_data.shape[0],nre), dtype=np.float32)
    v_mag.fill(np.nan)
    v_dir = np.empty((az_data.shape[0],nre), dtype=np.float32)
    v_dir.fill(np.nan)
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
            caz_data_cum = az_data[i,:,:]
            caz_data_cum = caz_data_cum[mask_data == 1]
            caz_data = (az_data[i,:,:] - az_data[i-1,:,:]) / delta_year
            caz_data = caz_data[mask_data == 1]
            crg_data = (rg_data[i,:,:] - rg_data[i-1,:,:]) / delta_year
            crg_data = crg_data[mask_data == 1]
            crg_data_cum = rg_data[i,:,:]
            crg_data_cum = crg_data_cum[mask_data == 1]
        delta_year_ar[i] = delta_year
        delta_day_ar[i] = delta_day
        rg_data_cum[i,:] = crg_data_cum
        az_data_cum[i,:] = caz_data_cum

    # perform linear interpolation of dx and dy offsets using delta_years
    tsamples_daily = np.arange(0, delta_day_ar.max()+1, 1) # samples every day for plotting
    tsamples_monthly = np.arange(0, np.cumsum(delta_day_ar).max()+1, 30) # for interpolation
    fit_dx = np.interp(x=tsamples_monthly, xp=np.cumsum(delta_day_ar), fp=rg_data_cum[:,1]) #using numpy - it's faster with numba
    fity = interp1d(timesteps.dt_start, np.cumsum(timesteps.dy), kind='linear')

    # use interpolated values to calculate velocity
    vel_magnitude = np.sqrt(caz_data**2 + crg_data**2)
    vel_direction = np.rad2deg(np.arctan2(caz_data, crg_data))
    v_mag[i,:] = vel_magnitude
    v_dir[i,:] = vel_direction

    # store mean and standard deviation into HDF file
    vdata = np.c_[np.nanmean(az_data_cum, axis=1), np.nanstd(az_data_cum, axis=1),
            np.nanmean(rg_data_cum, axis=1), np.nanstd(rg_data_cum, axis=1),
            np.nanmean(v_mag, axis=1), np.nanstd(v_mag, axis=1),
            np.nanpercentile(v_mag, axis=1, q=25), np.nanpercentile(v_mag, axis=1, q=25),
            np.nanmean(v_dir, axis=1), np.nanstd(v_dir, axis=1)]
    vel_mag_dir_df = pd.DataFrame(vdata, columns=['dy_mean_cum', 'dy_std_cum', 'dx_mean_cum', 'dx_std_cum', 'v_mean', 'v_std', 'v_25p', 'v_75p', 'dir_mean', 'dir_std'],
                                            index=[pd.to_datetime(dates, format='%Y%m%d')])
    vel_mag_dir_df.to_hdf(args.HDF_outfile, key='vel_mag_dir_df', complevel=7)

    fig, ax = plt.subplots(1,2, figsize = (20, 5), dpi=300)
    ax[0].errorbar(vel_mag_dir_df.index, vel_mag_dir_df.dx_mean_cum, yerr=vel_mag_dir_df.dx_std_cum)
    ax[0].plot(vel_mag_dir_df.index, vel_mag_dir_df.dx_mean_cum, 'k', lw=2)
    ax[0].set_ylim(-1,30)
    ax[0].grid()
    ax[0].set_ylabel("Cumulative displacement dx [pix]")
    ax[1].errorbar(vel_mag_dir_df.index, vel_mag_dir_df.dy_mean_cum, yerr=vel_mag_dir_df.dy_std_cum)
    ax[1].plot(vel_mag_dir_df.index, vel_mag_dir_df.dy_mean_cum, 'k', lw=2)
    ax[1].set_ylim(-1,30)
    ax[1].set_ylabel("Cumulative displacement dy [pix]")
    ax[1].grid()
    fig.suptitle('Cumulative TS from inversion with weights', fontsize=16)
    fig.tight_layout()
    fig.savefig(args.out_pngfname _ts.png')


    # prepare pcolormesh with histogram plot for velocity
    mintimedelta = 5 # [days]
    nbins = 100
    v = v_mag[1:,:]
    b = np.linspace(v.min(), v.max(), nbins+1)

    d = [datetime.datetime.strptime(s, '%Y%m%d') for s in dates]
    d = d[1:]
    ndays = (d[-1]-d[0]).days+1
    x = [d[0] + datetime.timedelta(days = i*mintimedelta) for i in range(ndays//mintimedelta+2)]
    img = np.nan * np.ones((nbins, ndays//mintimedelta+1))
    for j, di in enumerate(d):
        i = (di-d[0]).days//mintimedelta
        img[:, i], _ = np.histogram(v[j,:], b, density = True)

    fig, ax = plt.subplots(1, 3, figsize = (19.2, 10.8), dpi=300)
    im0 = ax[0].pcolormesh(x, b, img, norm = LogNorm(), cmap = plt.cm.magma_r)
    ax[0].plot(vel_mag_dir_df.index, vel_mag_dir_df.v_mean, 'k', lw=2)
    ax[0].set_ylim([0,400])
    cb0 = plt.colorbar(im0, ax=ax[0], location='bottom', pad=0.1)
    cb0.set_label('Probability')
    ax[0].set_title('Velocity from inversion with var weights')
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Velocity [m/y]')
    ax[0].grid()

    # prepare pcolormesh with histogram plot for dx cumulative
    v2 = rg_data_cum[1:,:]
    b = np.linspace(v2.min(), v2.max(), nbins+1)
    img2 = np.nan * np.ones((nbins, ndays//mintimedelta+1))
    for j, di in enumerate(d):
        i = (di-d[0]).days//mintimedelta
        img2[:, i], _ = np.histogram(v2[j,:], b, density = True)

    im1 = ax[1].pcolormesh(x, b, img2, norm = LogNorm(), cmap = plt.cm.viridis)
    ax[1].plot(vel_mag_dir_df.index, vel_mag_dir_df.dx_mean_cum, 'k', lw=2)
    ax[1].set_ylim([0,60])
    cb1 = plt.colorbar(im1, ax=ax[1], location='bottom', pad=0.1)
    cb1.set_label('Probability')
    ax[1].set_title('Cumulative dx from inversion with var weights')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('dx [m]')
    ax[1].grid()

    # prepare pcolormesh with histogram plot for dy cumulative
    v3 = az_data_cum[1:,:]
    b = np.linspace(v3.min(), v3.max(), nbins+1)
    img3 = np.nan * np.ones((nbins, ndays//mintimedelta+1))
    for j, di in enumerate(d):
        i = (di-d[0]).days//mintimedelta
        img3[:, i], _ = np.histogram(v3[j,:], b, density = True)

    im2 = ax[2].pcolormesh(x, b, img3, norm = LogNorm(), cmap = plt.cm.viridis)
    ax[2].plot(vel_mag_dir_df.index, vel_mag_dir_df.dy_mean_cum, 'k', lw=2)
    ax[2].set_ylim([0,60])
    cb2 = plt.colorbar(im2, ax=ax[2], location='bottom', pad=0.1)
    cb2.set_label('Probability')
    ax[2].set_title('Cumulative dy from inversion with var weights')
    ax[2].set_xlabel('Time')
    ax[2].set_ylabel('dy [m]')
    ax[2].grid()
    fig.tight_layout()
    fig.savefig('aoi6_velocity.png')
