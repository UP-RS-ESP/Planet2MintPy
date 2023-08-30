#############################################################
# Created by Bodo Bookhagen and Ariane Mueting, August 2023 #
#############################################################

# Make sure to install a proper conda environment
# conda create -n CC python=3.11 numpy scipy pandas matplotlib ipython cupy tqdm numba -c conda-forge
# pip install install numba-progress

import glob, os, csv, sys, subprocess, tqdm, gzip
from datetime import datetime

import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from numba import njit, prange
from numba_progress import ProgressBar

from scipy.interpolate import NearestNDInterpolator
from scipy import ndimage
from scipy import signal
from skimage import measure

@njit(parallel=True)
def angle_variance(directions):
    #calculate circular variance from stack of directions
    #scaled between 0 and 1
    cvar = np.empty((directions.shape[1],directions.shape[2]), dtype=np.float32)
    cvar.fill(np.nan)

    for i in prange(directions.shape[1]):
        for j in prange(directions.shape[2]):
            #iterate through all pixels and take variance in time
            deg = directions[:,i,j]
            deg = np.deg2rad(deg)
            deg = deg[~np.isnan(deg)]

            S = deg.copy()
            C = deg.copy()

            length = C.size

            S = np.sum(np.sin(S))
            C = np.sum(np.cos(C))
            R = np.sqrt(S**2 + C**2)
            R_avg = R/length
            V = 1 - R_avg

            cvar[i,j] = V
    return cvar


@njit(parallel=True)
def calc_angle_numba(dx_stack, dy_stack):
    # Calculate angle difference
    angle = np.empty(dx_stack.shape, dtype=np.float32)
    angle.fill(np.nan)

    for i in prange(dy_stack.shape[0]):
        dx_stackc = dx_stack[i,:,:].ravel()
        dy_stackc = dy_stack[i,:,:].ravel()
        dangle = np.rad2deg(np.arctan2(dy_stackc, dx_stackc))

        #convert to coordinates with North = 0
        dangle[dangle < 0] = np.abs(dangle[dangle < 0]) + 180.
        dangle = dangle - 90.
        dangle[dangle < 0] = np.abs(dangle[dangle < 0]) + 90.
        angle[i,:,:] = dangle.reshape(dx_stack.shape[1], dx_stack.shape[2])

    return angle


def circular_mean(angles, weights=None):
    # This code also allows for weighted averages.
    # It returns both the mean and the variance as defined
    # in the appropriate Wikipedia article. Calculations are in radians.
    # https://en.wikipedia.org/wiki/Circular_mean

    if weights is None:
        weights = np.ones(len(angles))
    vectors = [ [w*np.cos(a), w*np.sin(a)]  for a,w in zip(angles,weights) ]
    vector = np.sum(vectors, axis=0) / np.sum(weights)
    x,y = vector
    angle_mean = np.arctan2(y,x)
    angle_variance = 1. - np.linalg.norm(vector)  # x*2+y*2 = hypot(x,y)

    return angle_mean, angle_variance


@njit(parallel=True)
def variance_angle(deg):
    """
    Simplified variance of angle calculation
    deg: angles in degrees
    """
    deg = np.deg2rad(deg)
    deg = deg[~np.isnan(deg)]

    S = deg.copy()
    C = deg.copy()

    length = C.size

    S = np.sum(np.sin(S))
    C = np.sum(np.cos(C))
    R = np.sqrt(S**2 + C**2)
    R_avg = R/length
    V = 1- R_avg
    return V


@njit(parallel=True)
def NormalizeData(data):
    # Normalize data between 0 and 1 using min and max values
    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))


@njit(parallel=True)
def NormalizeData_p0298(data):
    # Normalize data between 0 and 1 using 2nd and 98th percentile
    return (data - np.nanpercentile(data, 2)) / (np.nanpercentile(data, 98) - np.nanpercentile(data, 2))


@njit(parallel=True)
def nanmedian_numba_ts(array_data):
    # nanmedian_numba - calculates nanmedian along time axis in parallel mode using numba
    # expects 3 dimensions:
    #  - dimension 0 contains time steps
    #  - dimension 1 and 2 contain image data
    nanmedian = np.empty((array_data.shape[0]), dtype=np.float32)
    nanmedian.fill(np.nan)
    for i in prange(array_data.shape[0]):
        nanmedian[i] = np.nanmedian(array_data[i, :, :])
    return nanmedian


@njit(parallel=True)
def nanmedian_numba(array_data):
    # nanmedian_numba - calculates nanmedian in parallel mode using numba
    # expects 3 dimensions:
    #  - dimension 0 contains time steps
    #  - dimension 1 and 2 contain image data
    nanmedian = np.empty((array_data.shape[1], array_data.shape[2]), dtype=np.float32)
    nanmedian.fill(np.nan)
    for i in prange(array_data.shape[1]):
        for j in prange(array_data.shape[2]):
            nanmedian[i, j] = np.nanmedian(array_data[:, i, j])
    return nanmedian


@njit(parallel=True)
def nanvar_numba(array_data):
    # nanvar_numba - calculates nanvar in parallel mode using numba
    # expects 3 dimensions:
    #  - dimension 0 contains time steps
    #  - dimension 1 and 2 contain image data
    nanvar = np.empty((array_data.shape[1], array_data.shape[2]), dtype=np.float32)
    nanvar.fill(np.nan)
    for i in prange(array_data.shape[1]):
        for j in prange(array_data.shape[2]):
            nanvar[i, j] = np.nanvar(array_data[:, i, j])
    return nanvar


@njit(parallel=True)
def nanstd_numba(array_data):
    # nanstd_numba - calculates nanstd in parallel mode using numba
    # expects 3 dimensions:
    #  - dimension 0 contains time steps
    #  - dimension 1 and 2 contain image data
    nanstd = np.empty((array_data.shape[1], array_data.shape[2]), dtype=np.float32)
    nanstd.fill(np.nan)
    for i in prange(array_data.shape[1]):
        for j in prange(array_data.shape[2]):
            nanstd[i, j] = np.nanstd(array_data[:, i, j])
    return nanstd


@njit(parallel=True)
def nanmean_numba(array_data):
    # nanmean_numba - calculates nanmean in parallel mode using numba
    # expects 3 dimensions:
    #  - dimension 0 contains time steps
    #  - dimension 1 and 2 contain image data
    nanmean = np.empty((array_data.shape[1], array_data.shape[2]), dtype=np.float32)
    nanmean.fill(np.nan)
    for i in prange(array_data.shape[1]):
        for j in prange(array_data.shape[2]):
            nanmean[i, j] = np.nanmean(array_data[:, i, j])
    return nanmean


@njit(parallel=True)
def mZscore_numba(array_data, array_median):
    # Modified zscore = 0.6745 * (x_data - median) / median_absolute_deviation

    # First calculate absolute difference followed by their median calculation
    ts_absdiff = np.empty(array_data.shape, dtype=np.float32)
    ts_absdiff.fill(np.nan)
    for i in prange(array_data.shape[0]):
        abs_diff = np.abs(array_data[i,:,:] - array_median)
        ts_absdiff[i,:,:] = abs_diff
    ts_absdiff = ts_absdiff.reshape((array_data.shape[0], array_data.shape[1]*array_data.shape[2]))
    median_abs_diff = np.empty(ts_absdiff.shape[1])
    for i in prange(ts_absdiff.shape[1]):
        median_abs_diff[i] = np.nanmedian(ts_absdiff[:,i])
    median_abs_diff = median_abs_diff.reshape(array_data.shape[1], array_data.shape[2])
#    median_abs_diff = np.nanmedian(ts_absdiff, axis=0)
    ts_absdiff = None

    # Calculate modified zScore
    ts_mZscore = np.empty(array_data.shape, dtype=np.float32)
    ts_mZscore.fill(np.nan)
    for i in prange(array_data.shape[0]):
        modified_zscore =  0.6745 * (array_data[i,:,:] - array_median) / median_abs_diff
        ts_mZscore[i,:,:] = modified_zscore

    return ts_mZscore


def mZscore(array_data, array_median):
    # Modified zscore = 0.6745 * (x_data - median) / median_absolute_deviation

    # First calculate absolute difference followed by their median calculation
    ts_absdiff = np.empty(array_data.shape, dtype=np.float32)
    ts_absdiff.fill(np.nan)
    for i in range(array_data.shape[0]):
        abs_diff = np.abs(array_data[i,:,:] - array_median)
        ts_absdiff[i,:,:] = abs_diff
    ts_absdiff = ts_absdiff.reshape((array_data.shape[0], array_data.shape[1]*array_data.shape[2]))
    median_abs_diff = np.empty(ts_absdiff.shape[1])
    for i in range(ts_absdiff.shape[1]):
        median_abs_diff[i] = np.nanmedian(ts_absdiff[:,i])
    median_abs_diff = median_abs_diff.reshape(array_data.shape[1], array_data.shape[2])
#    median_abs_diff = np.nanmedian(ts_absdiff, axis=0)
    ts_absdiff = None

    # Calculate modified zScore
    ts_mZscore = np.empty(array_data.shape, dtype=np.float32)
    ts_mZscore.fill(np.nan)
    for i in range(array_data.shape[0]):
        modified_zscore =  0.6745 * (array_data[i,:,:] - array_median) / median_abs_diff
        idxinf = np.where(np.isinf(modified_zscore))
        modified_zscore[idxinf] = np.nan
        ts_mZscore[i,:,:] = modified_zscore

    # normalize modified zScore with 2nd and 98th percentile of all data and then clip
    p98 = np.nanpercentile(ts_mZscore, 98)
    p02 = np.nanpercentile(ts_mZscore, 2)
    ts_mZscore = (ts_mZscore - p02)  / (p98 - p02)
    ts_mZscore = np.clip(ts_mZscore, 0, 1)

    return ts_mZscore


@njit(parallel=True)
def calc_dangle(dx_stack, dy_stack, dx_stack_median_ar, dy_stack_median_ar, mask_stack):
    # calculate angle difference between pixel and median-filter X and Y directions
    # the median filtered data are smoothed and
    ts_dangle = np.empty(dy_stack.shape, dtype=np.float32)
    ts_dangle.fill(np.nan)
    ts_dangle_median = np.empty(dy_stack.shape[0], dtype=np.float32)
    ts_dangle_median.fill(np.nan)
    ts_dangle_var = np.empty(dy_stack.shape[0], dtype=np.float32)
    ts_dangle_var.fill(np.nan)
    ts_dangle_mask = np.zeros(dy_stack.shape, dtype=np.int8)
    for i in prange(dy_stack.shape[0]):
        dx_stackc = dx_stack[i,:,:].ravel()
        dy_stackc = dy_stack[i,:,:].ravel()
        dx_stack_medianc = dx_stack_median_ar.ravel()
        dy_stack_medianc = dy_stack_median_ar.ravel()
        dangle = np.rad2deg(np.arctan2(dy_stackc-dy_stack_medianc, dx_stackc - dx_stack_medianc))
        dangle = np.clip(dangle, -90, 90)
        dangle_ar = dangle.reshape(dx_stack_median_ar.shape[0], dx_stack_median_ar.shape[1])
        mask_ar = np.where(np.abs(dangle_ar) == 90, 0, 1).astype(np.int8) # all angles of -90 and 90 are set to 0
        # make sure that all nans in mask_stack are also set to 0
        # need to work with ravel to make numba happy
        idxxy = np.where(mask_stack[i,:,:].ravel() == 0)[0].astype(np.int32)
        mask_ar.ravel()[idxxy] = np.int8(0)
        # set Nan border values to 0:
        idxxy = np.where(np.isnan(dangle))[0].astype(np.int32)
        mask_ar.ravel()[idxxy] = np.int8(0)
        ts_dangle_mask[i,:,:] = mask_ar
        ts_dangle[i,:,:] = np.cos(np.deg2rad(dangle_ar))
        ts_dangle_median[i] = np.nanmedian(ts_dangle[i,:,:])
        ts_dangle_var[i] = np.nanvar(ts_dangle[i,:,:])
        dangle = None
        dangle_ar = None
    return ts_dangle, ts_dangle_median, ts_dangle_var, ts_dangle_mask


@njit(parallel=True)
def calc_dangle_dem(dx_stack, dy_stack, dem_aspect):
    # calculate angle difference between pixel and median-filter X and Y directions
    # the median filtered data are smoothed and
    dem_dangle = np.empty(dy_stack.shape, dtype=np.float32)
    dem_dangle.fill(np.nan)
    dem_dangle_median = np.empty(dy_stack.shape[0], dtype=np.float32)
    dem_dangle_median.fill(np.nan)
    dem_dangle_var = np.empty(dy_stack.shape[0], dtype=np.float32)
    dem_dangle_var.fill(np.nan)
    dem_aspectc = dem_aspect.ravel()

    for i in prange(dy_stack.shape[0]):
        dx_stackc = dx_stack[i,:,:].ravel()
        dy_stackc = dy_stack[i,:,:].ravel()
        dangle = np.rad2deg(np.arctan2(dy_stackc, dx_stackc)) - dem_aspectc
        dangle = np.clip(dangle, -90, 90)
        dangle_ar = dangle.reshape(dem_aspect.shape[0], dem_aspect.shape[1])
        dem_dangle[i,:,:] = np.cos(np.deg2rad(dangle_ar.copy()))
        dem_dangle_median[i] = np.nanmedian(dem_dangle[i,:,:])
        dem_dangle_var[i] = np.nanvar(dem_dangle[i,:,:])
        dangle = None
        dangle_ar = None
    return dem_dangle, dem_dangle_median, dem_dangle_var


def median_gpu(array_stack, kernel_size=3):
    # only works if data has no NaN
    import cupy as cp
    from cupyx.scipy import ndimage as ndimagex

    # requires the installation of cupy and a GPU
    array_stack_gpu = cp.asarray(array_stack)
    array_stack_median_gpu = cp.empty(array_stack_gpu.shape, dtype=np.float32)
    array_stack_median_gpu.fill(np.nan)
    for i in range(array_stack_gpu.shape[0]):
        #apply 3x3 median filter for smoothed values
        array_stack_median_gpu[i,:,:] = ndimagex.median_filter(array_stack_gpu[i,:,:], size=kernel_size)
    array_stack = cp.asnumpy(array_stack_median_gpu)
    # clean up memory usage
    del array_stack_median_gpu
    del array_stack_gpu
    cp._default_memory_pool.free_all_blocks()
    return array_stack


def load_data(filelist, dxdy_size, output_path = 'npy', area_fname='DelMedio', mask=False, sensor='L3B'):
    from osgeo import gdal
    dx_npy_fname = os.path.join(output_path, area_fname + '_dx.npy.gz')
    dy_npy_fname = os.path.join(output_path, area_fname + '_dy.npy.gz')
    mask_npy_fname = os.path.join(output_path, area_fname + '_mask.npy.gz')
    deltay_npy_fname = os.path.join(output_path, area_fname + '_deltay.npy.gz')
    date0_npy_fname = os.path.join(output_path, area_fname + '_date0.npy.gz')
    date1_npy_fname = os.path.join(output_path, area_fname + '_date1.npy.gz')
    if os.path.exists(mask_npy_fname) and os.path.exists(dx_npy_fname) and os.path.exists(dy_npy_fname) and os.path.exists(deltay_npy_fname) and os.path.exists(date0_npy_fname) and os.path.exists(date1_npy_fname):
        print('gzipped npy files %s and %s exist'%(dx_npy_fname, dy_npy_fname) )
        return

    dx_stack = np.empty((len(filelist), dxdy_size[0], dxdy_size[1]), dtype=np.float32)
    dx_stack.fill(np.nan)
    dy_stack = np.empty((len(filelist), dxdy_size[0], dxdy_size[1]), dtype=np.float32)
    dy_stack.fill(np.nan)
    mask_stack = np.zeros((len(filelist), dxdy_size[0], dxdy_size[1]), dtype=np.int8)
    date0_stack = np.empty(len(filelist))
    date0_stack.fill(np.nan)
    date1_stack = np.empty(len(filelist))
    date1_stack.fill(np.nan)
    deltay_stack = np.empty(len(filelist))
    deltay_stack.fill(np.nan)


    print('Loading TIF files and storing to new array')
    for i in tqdm.tqdm(range(len(filelist))):
        # loop would benefit from parallelization
        cfile_basename = os.path.basename(filelist[i])
        cfile = filelist[i]
        ds = gdal.Open(cfile)
        gt = ds.GetGeoTransform()
        sr = ds.GetProjection()
        dx = ds.GetRasterBand(1).ReadAsArray()
        dy = ds.GetRasterBand(2).ReadAsArray()
        if mask == True:
            dxdy_mask = ds.GetRasterBand(3).ReadAsArray()
            dx[dxdy_mask == 0] = np.nan
            dy[dxdy_mask == 0] = np.nan
        else:
            #create mask with nan values from dx file
            dxdy_mask = np.ones(dx.shape)
            dxdy_mask[np.isnan(dx)] = 0
        #calculate date (time) difference in years from filename
        if sensor == 'L8':
            date0 = cfile_basename.split('_')[0]
            date1 = cfile_basename.split('_')[1]
        elif sensor == 'PS':
            date0 = cfile_basename.split('_')[0]
            #need to distinguish between PSBSD and PS2 scene IDs
            if len(cfile_basename.split('_')[3]) == 8:
                date1 = cfile_basename.split('_')[3]
            else:                 
                date1 = cfile_basename.split('_')[4]
        delta_days = datetime.strptime(date1, "%Y%m%d") - datetime.strptime(date0, "%Y%m%d")
        delta_year = delta_days.days/365
        dx = dx / delta_year #units are pixels per year
        dy = dy / delta_year #units are pixels per year
        deltay_stack[i] = delta_year
        date0_stack[i] = date0
        date1_stack[i] = date1

        dx_stack[i,:,:] = dx
        dy_stack[i,:,:] = dy
        mask_stack[i,:,:] = dxdy_mask
        ds = None

    if os.path.exists(dx_npy_fname) is False:
        print('saving %s to gzipped npy files'%dx_npy_fname)
        f = gzip.GzipFile(dx_npy_fname, "w")
        np.save(file=f, arr=dx_stack)
        f.close()
        f = None

    if os.path.exists(dy_npy_fname) is False:
        print('saving %s to gzipped npy files'%dy_npy_fname)
        f = gzip.GzipFile(dy_npy_fname, "w")
        np.save(file=f, arr=dy_stack)
        f.close()
        f = None

    if os.path.exists(mask_npy_fname) is False:
        print('saving %s to gzipped npy files'%mask_npy_fname)
        f = gzip.GzipFile(mask_npy_fname, "w")
        np.save(file=f, arr=mask_stack)
        f.close()
        f = None

    if os.path.exists(deltay_npy_fname) is False:
        print('saving %s to gzipped npy files'%deltay_npy_fname)
        f = gzip.GzipFile(deltay_npy_fname, "w")
        np.save(file=f, arr=deltay_stack)
        f.close()
        f = None

    if os.path.exists(date0_npy_fname) is False:
        print('saving %s to gzipped npy files'%date0_npy_fname)
        f = gzip.GzipFile(date0_npy_fname, "w")
        np.save(file=f, arr=date0_stack)
        f.close()
        f = None

    if os.path.exists(date1_npy_fname) is False:
        print('saving %s to gzipped npy files'%date1_npy_fname)
        f = gzip.GzipFile(date1_npy_fname, "w")
        np.save(file=f, arr=date1_stack)
        f.close()
        f = None


def load_tif_stacks(filelist, dxdy_size, mask=False):
    from osgeo import gdal
    dx_stack = np.empty((len(filelist), dxdy_size[0], dxdy_size[1]), dtype=np.float32)
    dx_stack.fill(np.nan)
    dy_stack = np.empty((len(filelist), dxdy_size[0], dxdy_size[1]), dtype=np.float32)
    dy_stack.fill(np.nan)
    mask_stack = np.zeros((len(filelist), dxdy_size[0], dxdy_size[1]), dtype=np.int8)

    for i in tqdm.tqdm(range(len(filelist))):
        # loop would benefit from parallelization
        cfile_basename = os.path.basename(filelist[i])
        cfile = filelist[i]
        ds = gdal.Open(cfile)
        dx = ds.GetRasterBand(1).ReadAsArray()
        dy = ds.GetRasterBand(2).ReadAsArray()
        if mask == True:
            dxdy_mask = ds.GetRasterBand(3).ReadAsArray()
            dx[dxdy_mask == 0] = np.nan
            dy[dxdy_mask == 0] = np.nan
        else:
            #create mask with nan values from dx file
            dxdy_mask = np.ones(dx.shape)
            dxdy_mask[np.isnan(dx)] = 0

        dx_stack[i,:,:] = dx
        dy_stack[i,:,:] = dy
        mask_stack[i,:,:] = dxdy_mask
        ds = None

    if mask == True:
        return dx_stack, dy_stack, mask_stack
    else:
        return dx_stack, dy_stack


def interp_nan_gpu(array_stack, interp_nan_fname):
    # RegularGridInterpolator - not working
    import cupy as cp
    # perform linear interpolation over each time step to remove NaN values
    # NaN values represent weak correlations or correlations that failed
    from cupyx.scipy.interpolate import RegularGridInterpolator
    interp_nan_fname = interp_nan_fname + '_nonan.npy.gz'

    array_stack_gpu = cp.asarray(array_stack)
    array_nonan_stack_gpu = cp.empty(array_stack.shape, dtype=np.float32)
    array_nonan_stack_gpu.fill(np.nan)
    x = np.arange(0,array_stack.shape[1])
    y = np.arange(0,array_stack.shape[2])
    print('Interpolating grid with NaNs and storing to new array')
    for i in tqdm.tqdm(range(array_stack_gpu.shape[0])):
        gridi = RegularGridInterpolator((x, y), array_stack[i,:,:],
            bounds_error=False, fill_value=np.nan, method='linear')
        array_nonan_stack_gpu[i,:,:] = cp.asarray(gridi.values)
        gridi=None
    array_nonan_stack = cp.asnumpy(array_nonan_stack_gpu)
    # clean up memory usage
    del array_nonan_stack_gpu
    del array_stack_gpu
    cp._default_memory_pool.free_all_blocks()

    if os.path.exists(interp_nan_fname) is False:
        print('saving to gzipped %s file'%interp_nan_fname)
        f = gzip.GzipFile(interp_nan_fname, "w")
        np.save(file=f, arr=array_nonan_stack)
        f.close()
        f = None

    return array_nonan_stack

def interp_nan(array_stack, interp_nan_fname):
    interp_nan_fname = interp_nan_fname + '_nonan.npy.gz'
    array_nonan_stack = np.empty(array_stack.shape, dtype=np.float32)
    array_nonan_stack.fill(np.nan)

    print('Interpolating grid with NaNs and storing to new array')
    for i in tqdm.tqdm(range(array_stack.shape[0])):
        array_stackc = array_stack[i,:,:] #units are pixels per year

        #interpolate NaN values and fill with nearest value. This step is required for median filtering
        #alternative is to use custom made median filter that ignores NaN values
        mask = np.where(~np.isnan(array_stackc))
        interp = NearestNDInterpolator(np.transpose(mask), array_stackc[mask])
        array_nonan_stack[i,:,:] = interp(*np.indices(array_stackc.shape))
        mask = None
        interp = None

    if os.path.exists(interp_nan_fname) is False:
        print('saving to gzipped %s file'%interp_nan_fname)
        f = gzip.GzipFile(interp_nan_fname, "w")
        np.save(file=f, arr=array_nonan_stack)
        f.close()
        f = None

    return array_nonan_stack


@njit(parallel=True)
def for_nanstddev(array_stack2, mask_stack, pad, shape, progress_hook):
    array_nanstddev = np.empty(shape, dtype=np.float32)
    array_nanstddev.fill(np.nan)
    for ts in prange(array_stack2.shape[0]):
        #iterate through each time step
        for i in prange(pad, array_stack2.shape[1]-pad-1):
            for j in prange(pad, array_stack2.shape[2]-pad-1):
                roi = array_stack2[ts, i - pad:i + pad + 1, j - pad:j + pad + 1]
                if mask_stack[ts, i - pad, j - pad] == 0:
                    k = np.nan
                elif np.all(np.isnan(roi)):
                    k = np.nan
                else:
                    # convolution or nanmedian
                    k = np.nanstd(roi)
                array_nanstddev[ts, i - pad, j - pad] = k
                progress_hook.update(1)
    return array_nanstddev


def filter2d_nanstddev(array_stack, mask_stack, kernel_size=7):
    pad = kernel_size // 2
    #
    # pad array with pad - only pad dimension 1 and 2
    # easier to pad than to check for border pixels
    # print('padding array')
    array_stack2 = np.pad(array_stack, ((0,0), (pad,pad), (pad,pad)), 'reflect')
    print('running nan std.dev. filtering')
    with ProgressBar(total=array_stack.shape[0]*array_stack.shape[1]*array_stack.shape[2]) as progress:
        array_nanstddev = for_nanstddev(array_stack2, mask_stack, pad, array_stack.shape, progress)

    return array_nanstddev


@njit(parallel=True)
def for_nanmedian(array_stack2, mask_stack, pad, shape, progress_hook):
    array_nanmedian = np.empty(shape, dtype=np.float32)
    array_nanmedian.fill(np.nan)
    for ts in prange(array_stack2.shape[0]):
        #iterate through each time step
        for i in prange(pad, array_stack2.shape[1]-pad-1):
            for j in prange(pad, array_stack2.shape[2]-pad-1):
                roi = array_stack2[ts, i - pad:i + pad + 1, j - pad:j + pad + 1]
                if mask_stack[ts, i - pad, j - pad] == 0:
                    k = np.nan
                elif np.all(np.isnan(roi)):
                    k = np.nan
                else:
                    # convolution or nanmedian
                    k = np.nanmedian(roi)
                array_nanmedian[ts, i - pad, j - pad] = k
                progress_hook.update(1)
    return array_nanmedian


def filter2d_nanmedian(array_stack, mask_stack, kernel_size=7):
    pad = kernel_size // 2
    #
    # pad array with pad - only pad dimension 1 and 2
    # easier to pad than to check for border pixels
    # print('padding array')
    array_stack2 = np.pad(array_stack, ((0,0), (pad,pad), (pad,pad)), 'reflect')
    print('running nan median filtering')
    with ProgressBar(total=array_stack.shape[0]*array_stack.shape[1]*array_stack.shape[2]) as progress:
        array_nanmedian = for_nanmedian(array_stack2, mask_stack, pad, array_stack.shape, progress)

    return array_nanmedian


@njit(parallel=True)
def for_nanmedian_nomask(array_stack2, pad, shape, progress_hook):
    array_nanmedian = np.empty(shape, dtype=np.float32)
    array_nanmedian.fill(np.nan)
    for ts in prange(array_stack2.shape[0]):
        #iterate through each time step
        for i in prange(pad, array_stack2.shape[1]-pad-1):
            for j in prange(pad, array_stack2.shape[2]-pad-1):
                roi = array_stack2[ts, i - pad:i + pad + 1, j - pad:j + pad + 1]
                if array_stack2[ts, i - pad, j - pad] == 0:
                    k = np.nan
                elif np.all(np.isnan(roi)):
                    k = np.nan
                else:
                    # convolution or nanmedian
                    k = np.nanmedian(roi)
                array_nanmedian[ts, i - pad, j - pad] = k
                progress_hook.update(1)
    return array_nanmedian


def filter2d_nanmedian_nomask(array_stack, kernel_size=7):
    pad = kernel_size // 2
    #
    # pad array with pad - only pad dimension 1 and 2
    # easier to pad than to check for border pixels
    # print('padding array')
    array_stack2 = np.pad(array_stack, ((0,0), (pad,pad), (pad,pad)), 'reflect')
    print('running nan median filtering')
    with ProgressBar(total=array_stack.shape[0]*array_stack.shape[1]*array_stack.shape[2]) as progress:
        array_nanmedian = for_nanmedian_nomask(array_stack2, pad, array_stack.shape, progress)

    return array_nanmedian


def write_Geotiff(input_tif, array, output_tif):
    from osgeo import gdal
    #load georeference information from existing tif file
    ds = gdal.Open(input_tif)
    gt = ds.GetGeoTransform()
    sr = ds.GetProjection()

    #set nan to -9999
    array[np.isnan(array)] = -9999.
    # write to file
    driver = gdal.GetDriverByName('GTiff')
    ds_write = driver.Create(output_tif, xsize=ds.RasterXSize,
                         ysize=ds.RasterYSize, bands=1,
                         eType=gdal.GDT_Float32,
                         options=['COMPRESS=LZW', 'ZLEVEL=7']
                         )
    ds_write.GetRasterBand(1).WriteArray(array)
    ds_write.GetRasterBand(1).SetNoDataValue(-9999.)
    # Setup projection and geo-transform
    ds_write.SetProjection(sr)
    ds_write.SetGeoTransform(gt)
    # Save and close the file
    ds_write.FlushCache()
    driver = None
    ds_write = None
    ds = None

    return


def write_Geotiff_ts(input_tif, array_ts, date0_stack, date1_stack, output_prefix, output_postfix, output_dir):
    from osgeo import gdal
    #load georeference information from existing tif file
    ds = gdal.Open(input_tif)
    gt = ds.GetGeoTransform()
    sr = ds.GetProjection()
    print('Writing time series tif files to %s'%output_dir)
    for i in tqdm.tqdm(range(array_ts.shape[0])):
        output_tif = '%s_%s_%s_%s.tif'%(output_prefix, int(date0_stack[i]), int(date1_stack[i]), output_postfix)
        output_tif = os.path.join(output_dir, output_tif)

        array = array_ts[i,:,:]
        #set nan to -9999
        array[np.isnan(array)] = -9999.
        # write to file
        driver = gdal.GetDriverByName('GTiff')
        ds_write = driver.Create(output_tif, xsize=ds.RasterXSize,
                             ysize=ds.RasterYSize, bands=1,
                             eType=gdal.GDT_Float32,
                             options=['COMPRESS=LZW', 'ZLEVEL=7']
                             )
        ds_write.GetRasterBand(1).WriteArray(array)
        ds_write.GetRasterBand(1).SetNoDataValue(-9999.)
        # Setup projection and geo-transform
        ds_write.SetProjection(sr)
        ds_write.SetGeoTransform(gt)
        # Save the file
        ds_write.FlushCache()
        ds_write = None
        driver = None
    ds = None

    return


def write_Geotiff_ts_mask(input_tif, array_ts, date0_stack, date1_stack, output_prefix, output_postfix, output_dir):
    from osgeo import gdal
    #load georeference information from existing tif file
    ds = gdal.Open(input_tif)
    gt = ds.GetGeoTransform()
    sr = ds.GetProjection()
    print('Writing mask time series file to %s'%output_dir)
    for i in tqdm.tqdm(range(date0_stack.shape[0])):
        output_tif = '%s_%s_%s_%s.tif'%(output_prefix, int(date0_stack[i]), int(date1_stack[i]), output_postfix)
        output_tif = os.path.join(output_dir, output_tif)

        array = array_ts
        #set nan to -9999
        array[np.isnan(array)] = -9999.
        # write to file
        driver = gdal.GetDriverByName('GTiff')
        ds_write = driver.Create(output_tif, xsize=ds.RasterXSize,
                             ysize=ds.RasterYSize, bands=1,
                             eType=gdal.GDT_Float32,
                             options=['COMPRESS=LZW', 'ZLEVEL=7']
                             )
        ds_write.GetRasterBand(1).WriteArray(array)
        ds_write.GetRasterBand(1).SetNoDataValue(-9999.)
        # Setup projection and geo-transform
        ds_write.SetProjection(sr)
        ds_write.SetGeoTransform(gt)
        # Save the file
        ds_write.FlushCache()
        ds_write = None
        driver = None
    ds = None

    return


def plot_dxdy_median(dx_stack_median_ar, dy_stack_median_ar, dx_stack_var_ar, dy_stack_var_ar, nre, stack_median_var_4plots_fname):
    # plot median and variance of all median-filtered time slices
    fig, ax = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    vmin_lb = np.round(np.nanpercentile(dx_stack_median_ar,2),2)
    vmax_ub = np.abs(vmin_lb)
    im0 = ax[0,0].imshow(dx_stack_median_ar, cmap='Spectral', vmin=vmin_lb, vmax=vmax_ub)
    ax[0,0].set_title('Dx median stack of 7x7 median filter (n=%d)'%nre, fontsize=14)
    cb0 = fig.colorbar(im0, ax=ax[0,0], location='bottom', pad=0.1)
    cb0.set_label('dx offset (px)')
    vmin_lb = np.round(np.nanpercentile(dy_stack_median_ar,2),2)
    vmax_ub = np.abs(vmin_lb)
    im1 = ax[0,1].imshow(dy_stack_median_ar, cmap='Spectral', vmin=vmin_lb, vmax=vmax_ub)
    ax[0,1].set_title('Dy median stack of 7x7 median filter (n=%d)'%nre, fontsize=14)
    cb1 = fig.colorbar(im1, ax=ax[0,1], location='bottom', pad=0.1)
    cb1.set_label('dy offset (px)')
    im2 = ax[1,0].imshow(dx_stack_var_ar, cmap='viridis', vmin=0, vmax=0.1)
    ax[1,0].set_title('Dx variance stack of 7x7 median filter (n=%d)'%nre, fontsize=14)
    cb2 = fig.colorbar(im2, ax=ax[1,0], location='bottom', pad=0.1)
    cb2.set_label('dx offset variance (px)')
    im3 = ax[1,1].imshow(dy_stack_var_ar, cmap='viridis', vmin=0, vmax=0.1)
    ax[1,1].set_title('Dy variance stack of 7x7 median filter (n=%d)'%nre, fontsize=14)
    cb3 = fig.colorbar(im3, ax=ax[1,1], location='bottom', pad=0.1)
    cb3.set_label('dy offset variance (px)')
    fig.tight_layout()
    fig.savefig(stack_median_var_4plots_fname)


def plot_mask_sum(ts_dangle_mask_sum, nre, masksum_fname):
    # sum of all mask pixels
    ts_dangle_2plot = np.float32(ts_dangle_mask_sum)
    ts_dangle_2plot[ts_dangle_2plot == 0] = np.nan
    fig, ax = plt.subplots(1, 2, figsize=(16, 8), dpi=300)
    im0 = ax[0].imshow(ts_dangle_2plot, cmap='viridis')
    ax[0].set_title('Sum of good values (lower values are more masked values) (n=%d)'%nre, fontsize=14)
    cb0 = fig.colorbar(im0, ax=ax[0], location='bottom', pad=0.1)
    cb0.set_label('nr. of measurements')
    im1 = ax[1].imshow(ts_dangle_2plot/nre, cmap='magma')
    ax[1].set_title('Fraction of timesteps with data (1=all data) (n=%d)'%nre, fontsize=14)
    cb1 = fig.colorbar(im1, ax=ax[1], location='bottom', pad=0.1)
    cb1.set_label('data percentage (%)')
    fig.tight_layout()
    fig.savefig(masksum_fname)


def plot_direction_sd_mask(directions_sd, mask, nre, directions_sd_fname):
    # plotting standard deviation
    fig, ax = plt.subplots(1, 2, figsize=(16, 8), dpi=300)
    im0 = ax[0].imshow(directions_sd, cmap='viridis', vmin=0, vmax=90)
    cb0 = plt.colorbar(im0, ax=ax[0], location='bottom', pad=0.1)
    cb0.set_label('Std. Dev. Directions (degree)')
    ax[0].set_title('Standard deviation of directions through time (n=%d)'%nre, fontsize=14)

    im1 = ax[1].imshow(mask, cmap='gray_r')
    ax[1].set_title('Mask based on direction SD (n=%d)'%nre, fontsize=14)
    cb1 = fig.colorbar(im1, ax=ax[1], location='bottom', pad=0.1)
    cb1.set_label('mask')
    fig.tight_layout()
    fig.savefig(directions_sd_fname)


def plot_direction_magnitude(direction_stack_median, magnitude_stack_median, direction_stack_var, magnitude_stack_var, nre, stack_median_direction_magntitude_4plots_fname):
    fig, ax = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    im0 = ax[0,0].imshow(direction_stack_median, cmap='rainbow', vmin=-90, vmax=90)
    ax[0,0].set_title('Velocity Direction (n=%d)'%nre, fontsize=14)
    cb0 = fig.colorbar(im0, ax=ax[0,0], location='bottom', pad=0.1)
    cb0.set_label('Direction (degree)')
    im1 = ax[0,1].imshow(magnitude_stack_median, cmap='magma', vmin=0, vmax=0.3)
    ax[0,1].set_title('Velocity Magnitude (n=%d)'%nre, fontsize=14)
    cb1 = fig.colorbar(im1, ax=ax[0,1], location='bottom', pad=0.1)
    cb1.set_label('Magnitude (px/y)')
    im2 = ax[1,0].imshow(direction_stack_var, cmap='viridis', vmin=0, vmax=45)
    ax[1,0].set_title('Velocity Direction variance (n=%d)'%nre, fontsize=14)
    cb2 = fig.colorbar(im2, ax=ax[1,0], location='bottom', pad=0.1)
    cb2.set_label('Direction (degree)')
    im3 = ax[1,1].imshow(magnitude_stack_var, cmap='viridis', vmin=0, vmax=0.1)
    ax[1,1].set_title('Velocity Magnitude variance (n=%d)'%nre, fontsize=14)
    cb3 = fig.colorbar(im3, ax=ax[1,1], location='bottom', pad=0.1)
    cb3.set_label('Magnitude (px/y)')
    fig.tight_layout()
    fig.savefig(stack_median_direction_magntitude_4plots_fname)


def plot_direction_magnitude_ts_dem(direction_stack_median, magnitude_stack_median, direction_stack_var, magnitude_stack_var, nre, stack_median_direction_magntitude_4plots_fname):
#not finished yet
    fig, ax = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    im0 = ax[0,0].imshow(direction_stack_median, cmap='rainbow', vmin=-90, vmax=90)
    ax[0,0].set_title('Velocity Direction (n=%d)'%nre, fontsize=14)
    cb0 = fig.colorbar(im0, ax=ax[0,0], location='bottom', pad=0.1)
    cb0.set_label('Direction (degree)')
    im1 = ax[0,1].imshow(magnitude_stack_median, cmap='magma', vmin=0, vmax=0.3)
    ax[0,1].set_title('Velocity Magnitude (n=%d)'%nre, fontsize=14)
    cb1 = fig.colorbar(im1, ax=ax[0,1], location='bottom', pad=0.1)
    cb1.set_label('Magnitude (px/y)')
    im2 = ax[1,0].imshow(direction_stack_var, cmap='viridis', vmin=0, vmax=50)
    ax[1,0].set_title('Velocity Direction variance (n=%d)'%nre, fontsize=14)
    cb2 = fig.colorbar(im2, ax=ax[1,0], location='bottom', pad=0.1)
    cb2.set_label('Direction (degree)')
    im3 = ax[1,1].imshow(magnitude_stack_var, cmap='viridis', vmin=0, vmax=0.1)
    ax[1,1].set_title('Velocity Magnitude variance (n=%d)'%nre, fontsize=14)
    cb3 = fig.colorbar(im3, ax=ax[1,1], location='bottom', pad=0.1)
    cb3.set_label('Magnitude (px/y)')
    fig.tight_layout()
    fig.savefig(stack_median_direction_magntitude_4plots_fname)


def plot_2example_4metrics(imin, imax, ts_dangle, dx_mZscore, dy_mZscore, combined_score, date0_stack, date1_stack, deltay_stack,combined_scores_min_max_fname):
    # plotting individual dates for 2 example (min/max quality)
    fig, ax = plt.subplots(2, 4, figsize=(20, 12), dpi=300)

    im0 = ax[0,0].imshow(ts_dangle[imax,:,:], cmap='viridis', vmin=0, vmax=1)
    ax[0,0].set_title('MAX cos($\Delta$Angle) ID: %d (%s - %s, $\Delta$y=%1.1f)'%(imax,
        datetime.strftime(datetime.strptime(str(int(date0_stack[imax])), "%Y%m%d"), "%Y-%m-%d"),
        datetime.strftime(datetime.strptime(str(int(date1_stack[imax])), "%Y%m%d"), "%Y-%m-%d"),
        deltay_stack[imax]), fontsize=12)
    cb0 = fig.colorbar(im0, ax=ax[0,0], location='bottom', pad=0.1)
    cb0.set_label('Confidence [0,1]')
    im0b = ax[0,1].imshow(dx_mZscore[imax,:,:], cmap='cividis', vmin=0, vmax=1)
    ax[0,1].set_title('MAX dx_mZscore', fontsize=12)# ID: %d (%s - %s, $\Delta$y=%1.1f)'%(i,
    #    datetime.strftime(datetime.strptime(str(int(date0_stack[i])), "%Y%m%d"), "%Y-%m-%d"),
    #    datetime.strftime(datetime.strptime(str(int(date1_stack[i])), "%Y%m%d"), "%Y-%m-%d"),
    #    deltay_stack[i]), fontsize=12)
    cb0b = fig.colorbar(im0b, ax=ax[0,1], location='bottom', pad=0.1)
    cb0b.set_label('dx mZscore [0,1]')
    im0c = ax[0,2].imshow(dy_mZscore[imax,:,:], cmap='cividis', vmin=0, vmax=1)
    ax[0,2].set_title('MAX dy_mZscore', fontsize=12)# ID: %d (%s - %s, $\Delta$y=%1.1f)'%(i,
    #    datetime.strftime(datetime.strptime(str(int(date0_stack[i])), "%Y%m%d"), "%Y-%m-%d"),
    #    datetime.strftime(datetime.strptime(str(int(date1_stack[i])), "%Y%m%d"), "%Y-%m-%d"),
    #    deltay_stack[i]),fontsize=12)
    cb0c = fig.colorbar(im0c, ax=ax[0,2], location='bottom', pad=0.1)
    cb0c.set_label('dy mZscore [0,1]')
    im0d = ax[0,3].imshow(combined_score[imax,:,:], cmap='plasma', vmin=0, vmax=1)
    ax[0,3].set_title('MAX combined', fontsize=12) #ID: %d (%s - %s, $\Delta$y=%1.1f)'%(i,
    #    datetime.strftime(datetime.strptime(str(int(date0_stack[i])), "%Y%m%d"), "%Y-%m-%d"),
    #    datetime.strftime(datetime.strptime(str(int(date1_stack[i])), "%Y%m%d"), "%Y-%m-%d"),
    #    deltay_stack[i]),fontsize=12)
    cb0d = fig.colorbar(im0d, ax=ax[0,3], location='bottom', pad=0.1)
    cb0d.set_label('combined [0,1]')

    im1 = ax[1,0].imshow(ts_dangle[imin,:,:], cmap='viridis', vmin=0, vmax=1)
    ax[1,0].set_title('MIN cos($\Delta$Angle) ID: %d (%s - %s, $\Delta$y=%1.1f)'%(imin,
        datetime.strftime(datetime.strptime(str(int(date0_stack[imin])), "%Y%m%d"), "%Y-%m-%d"),
        datetime.strftime(datetime.strptime(str(int(date1_stack[imin])), "%Y%m%d"), "%Y-%m-%d"),
        deltay_stack[imin]),fontsize=12)
    cb1 = fig.colorbar(im1, ax=ax[1,0], location='bottom', pad=0.1)
    cb1.set_label('Confidence [0,1]')
    im1b = ax[1,1].imshow(dx_mZscore[imin,:,:], cmap='cividis', vmin=0, vmax=1)
    ax[1,1].set_title('MIN dx_mZscore', fontsize=12)# ID: %d (%s - %s, $\Delta$y=%1.1f)'%(i,
    #    datetime.strftime(datetime.strptime(str(int(date0_stack[i])), "%Y%m%d"), "%Y-%m-%d"),
    #    datetime.strftime(datetime.strptime(str(int(date1_stack[i])), "%Y%m%d"), "%Y-%m-%d"),
    #    deltay_stack[i]),fontsize=12)
    cb1b = fig.colorbar(im1b, ax=ax[1,1], location='bottom', pad=0.1)
    cb1b.set_label('dx mZscore [0,1]')
    im1c = ax[1,2].imshow(dy_mZscore[imin,:,:], cmap='cividis', vmin=0, vmax=1)
    ax[1,2].set_title('MIN dy_mZscore', fontsize=12) # ID: %d (%s - %s, $\Delta$y=%1.1f)'%(i,
    #    datetime.strftime(datetime.strptime(str(int(date0_stack[i])), "%Y%m%d"), "%Y-%m-%d"),
    #    datetime.strftime(datetime.strptime(str(int(date1_stack[i])), "%Y%m%d"), "%Y-%m-%d"),
    #    deltay_stack[i]),fontsize=12)
    cb1c = fig.colorbar(im1c, ax=ax[1,2], location='bottom', pad=0.1)
    cb1c.set_label('dy mZscore [0,1]')
    im1d = ax[1,3].imshow(ts_dangle[imin,:,:], cmap='plasma', vmin=0, vmax=1)
    ax[1,3].set_title('MIN combined', fontsize=12)# ID: %d (%s - %s, $\Delta$y=%1.1f)'%(i,
    #    datetime.strftime(datetime.strptime(str(int(date0_stack[i])), "%Y%m%d"), "%Y-%m-%d"),
    #    datetime.strftime(datetime.strptime(str(int(date1_stack[i])), "%Y%m%d"), "%Y-%m-%d"),
    #    deltay_stack[i]),fontsize=12)
    cb1d = fig.colorbar(im1d, ax=ax[1,3], location='bottom', pad=0.1)
    cb1d.set_label('combined [0,1]')
    fig.tight_layout()
    fig.savefig(combined_scores_min_max_fname)


def plot_2example_2metrics(imin, imax, ts_dangle, combined_score, date0_stack, date1_stack, deltay_stack,combined_scores_min_max_fname):
    # plotting individual dates for 2 example (min/max quality)
    fig, ax = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

    im0 = ax[0,0].imshow(ts_dangle[imax,:,:], cmap='viridis', vmin=0, vmax=1)
    ax[0,0].set_title('MAX cos($\Delta$Angle) ID: %d (%s - %s, $\Delta$y=%1.1f)'%(imax,
        datetime.strftime(datetime.strptime(str(int(date0_stack[imax])), "%Y%m%d"), "%Y-%m-%d"),
        datetime.strftime(datetime.strptime(str(int(date1_stack[imax])), "%Y%m%d"), "%Y-%m-%d"),
        deltay_stack[imax]), fontsize=12)
    cb0 = fig.colorbar(im0, ax=ax[0,0], location='bottom', pad=0.1)
    cb0.set_label('Confidence [0,1]')
    im0d = ax[0,1].imshow(combined_score[imax,:,:], cmap='plasma', vmin=0, vmax=1)
    ax[0,1].set_title('MAX combined', fontsize=12) #ID: %d (%s - %s, $\Delta$y=%1.1f)'%(i,
    #    datetime.strftime(datetime.strptime(str(int(date0_stack[i])), "%Y%m%d"), "%Y-%m-%d"),
    #    datetime.strftime(datetime.strptime(str(int(date1_stack[i])), "%Y%m%d"), "%Y-%m-%d"),
    #    deltay_stack[i]),fontsize=12)
    cb0d = fig.colorbar(im0d, ax=ax[0,1], location='bottom', pad=0.1)
    cb0d.set_label('combined [0,1]')

    im1 = ax[1,0].imshow(ts_dangle[imin,:,:], cmap='viridis', vmin=0, vmax=1)
    ax[1,0].set_title('MIN cos($\Delta$Angle) ID: %d (%s - %s, $\Delta$y=%1.1f)'%(imin,
        datetime.strftime(datetime.strptime(str(int(date0_stack[imin])), "%Y%m%d"), "%Y-%m-%d"),
        datetime.strftime(datetime.strptime(str(int(date1_stack[imin])), "%Y%m%d"), "%Y-%m-%d"),
        deltay_stack[imin]),fontsize=12)
    cb1 = fig.colorbar(im1, ax=ax[1,0], location='bottom', pad=0.1)
    cb1.set_label('Confidence [0,1]')
    im1d = ax[1,1].imshow(ts_dangle[imin,:,:], cmap='plasma', vmin=0, vmax=1)
    ax[1,1].set_title('MIN combined', fontsize=12)# ID: %d (%s - %s, $\Delta$y=%1.1f)'%(i,
    #    datetime.strftime(datetime.strptime(str(int(date0_stack[i])), "%Y%m%d"), "%Y-%m-%d"),
    #    datetime.strftime(datetime.strptime(str(int(date1_stack[i])), "%Y%m%d"), "%Y-%m-%d"),
    #    deltay_stack[i]),fontsize=12)
    cb1d = fig.colorbar(im1d, ax=ax[1,1], location='bottom', pad=0.1)
    cb1d.set_label('combined [0,1]')
    fig.tight_layout()
    fig.savefig(combined_scores_min_max_fname)


def plot_2example_3metrics(imin, imax, ts_dangle, combined_score, ts_dangle_mask, date0_stack, date1_stack, deltay_stack,combined_scores_min_max_fname):
    # plotting individual dates for 2 example (min/max quality)
    fig, ax = plt.subplots(2, 3, figsize=(16, 12), dpi=300)

    im0 = ax[0,0].imshow(ts_dangle[imax,:,:], cmap='viridis', vmin=0, vmax=1)
    ax[0,0].set_title('MAX cos($\Delta$Angle) ID: %d (%s - %s, $\Delta$y=%1.1f)'%(imax,
        datetime.strftime(datetime.strptime(str(int(date0_stack[imax])), "%Y%m%d"), "%Y-%m-%d"),
        datetime.strftime(datetime.strptime(str(int(date1_stack[imax])), "%Y%m%d"), "%Y-%m-%d"),
        deltay_stack[imax]), fontsize=12)
    cb0 = fig.colorbar(im0, ax=ax[0,0], location='bottom', pad=0.1)
    cb0.set_label('confidence [0,1]')
    im0d = ax[0,1].imshow(combined_score[imax,:,:], cmap='plasma', vmin=0, vmax=1)
    ax[0,1].set_title('MAX combined', fontsize=12) #ID: %d (%s - %s, $\Delta$y=%1.1f)'%(i,
    cb0d = fig.colorbar(im0d, ax=ax[0,1], location='bottom', pad=0.1)
    cb0d.set_label('combined [0,1]')
    im0e = ax[0,2].imshow(ts_dangle_mask[imax,:,:], cmap='gray_r', vmin=0, vmax=1)
    ax[0,2].set_title('Good value mask (1==True or good value)', fontsize=12) #ID: %d (%s - %s, $\Delta$y=%1.1f)'%(i,
    cb0e = fig.colorbar(im0e, ax=ax[0,2], location='bottom', pad=0.1)
    cb0e.set_label('mask')

    im1 = ax[1,0].imshow(ts_dangle[imin,:,:], cmap='viridis', vmin=0, vmax=1)
    ax[1,0].set_title('MIN cos($\Delta$Angle) ID: %d (%s - %s, $\Delta$y=%1.1f)'%(imin,
        datetime.strftime(datetime.strptime(str(int(date0_stack[imin])), "%Y%m%d"), "%Y-%m-%d"),
        datetime.strftime(datetime.strptime(str(int(date1_stack[imin])), "%Y%m%d"), "%Y-%m-%d"),
        deltay_stack[imin]),fontsize=12)
    cb1 = fig.colorbar(im1, ax=ax[1,0], location='bottom', pad=0.1)
    cb1.set_label('confidence [0,1]')
    im1d = ax[1,1].imshow(ts_dangle[imin,:,:], cmap='plasma', vmin=0, vmax=1)
    ax[1,1].set_title('MIN combined', fontsize=12)# ID: %d (%s - %s, $\Delta$y=%1.1f)'%(i,
    cb1d = fig.colorbar(im1d, ax=ax[1,1], location='bottom', pad=0.1)
    cb1d.set_label('combined [0,1]')
    im1e = ax[1,2].imshow(ts_dangle_mask[imin,:,:], cmap='gray_r', vmin=0, vmax=1)
    ax[1,2].set_title('Good value mask (1==True or good value)', fontsize=12)# ID: %d (%s - %s, $\Delta$y=%1.1f)'%(i,
    cb1d = fig.colorbar(im1e, ax=ax[1,2], location='bottom', pad=0.1)
    cb1d.set_label('mask')
    fig.tight_layout()
    fig.savefig(combined_scores_min_max_fname)


def aspect_slope_dem(dem_fname, aspect_out_fname, slope_out_fname, kernel_size = 9):
    # Takes DEM (will need to be same size as correlation images - same nrows and ncols)
    # calculates aspect and slope within window given by kernel_size.
    from osgeo import gdal
    #load georeference information from existing tif file
    ds = gdal.Open(dem_fname)
    gt = ds.GetGeoTransform()
    sr = ds.GetProjection()
    dem = np.array(ds.GetRasterBand(1).ReadAsArray())
    pad = kernel_size // 2
    f_element = np.arange(-pad, pad+1, 1, dtype=int)
    xf = np.tile(f_element, (kernel_size,1))
    dem_x = signal.convolve2d(dem, xf, boundary='symm', mode='same')
    dem_y = signal.convolve2d(dem, xf.transpose(), boundary='symm', mode='same')
    # both units will be returned in degree
    slope = np.rad2deg(np.arctan(np.sqrt(dem_y**2+dem_x**2) / gt[1]))
    aspect = np.rad2deg(np.arctan2(dem_y, dem_x))

    #convert to coordinates with North = 0
    aspect.ravel()[aspect.ravel() < 0] = np.abs(aspect.ravel()[aspect.ravel() < 0]) + 180.
    aspect = aspect - 90.
    aspect.ravel()[aspect.ravel() < 0] = np.abs(aspect.ravel()[aspect.ravel() < 0]) + 90.

    #write SLOPE to geotiff
    #set nan to -9999
    slope[np.isnan(slope)] = -9999.
    # write to file
    driver = gdal.GetDriverByName('GTiff')
    ds_write = driver.Create(slope_out_fname, xsize=ds.RasterXSize,
                         ysize=ds.RasterYSize, bands=1,
                         eType=gdal.GDT_Float32,
                         options=['COMPRESS=LZW', 'ZLEVEL=7']
                         )
    ds_write.GetRasterBand(1).WriteArray(slope)
    ds_write.GetRasterBand(1).SetNoDataValue(-9999.)
    ds_write.SetProjection(sr)
    ds_write.SetGeoTransform(gt)
    ds_write.FlushCache()
    ds_write = None

    #write ASPECT to geotiff
    #set nan to -9999
    aspect[np.isnan(aspect)] = -9999.
    driver = None
    driver = gdal.GetDriverByName('GTiff')
    ds_write = driver.Create(aspect_out_fname, xsize=ds.RasterXSize,
                         ysize=ds.RasterYSize, bands=1,
                         eType=gdal.GDT_Float32,
                         options=['COMPRESS=LZW', 'ZLEVEL=7']
                         )
    ds_write.GetRasterBand(1).WriteArray(aspect)
    ds_write.GetRasterBand(1).SetNoDataValue(-9999.)
    ds_write.SetProjection(sr)
    ds_write.SetGeoTransform(gt)
    ds_write.FlushCache()
    ds_write = None
    driver = None
    ds = None

    return slope, aspect

def read_file(file, b=1):
    with rasterio.open(file) as src:
        return(src.read(b))

def calc_direction(fn):
    with rasterio.open(fn) as src:
        # get raster resolution from metadata
        meta = src.meta

        # first band is offset in x direction, second band in y
        dx = src.read(1)
        dy = src.read(2)

        if meta["count"] == 3:
           # print("Interpreting the third band as good pixel mask.")
            valid = src.read(3)
            dx[valid == 0] = np.nan
            dy[valid == 0] = np.nan

    #calculate angle to north
    north = np.array([0,1])
    #stack x and y offset to have a 3d array with vectors along axis 2
    vector_2 = np.dstack((dx,dy))
    unit_vector_1 = north / np.linalg.norm(north)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2, axis = 2, keepdims = True)
    #there np.tensordot is needed (instead of np.dot) because of the multiple dimensions of the input arrays
    dot_product = np.tensordot(unit_vector_1, unit_vector_2, axes=([0],[2]))
    direction = np.rad2deg(np.arccos(dot_product))

    #as always the smallest angle to north is given, values need to be substracted from 360 if x is negative
    subtract = np.zeros(dx.shape)
    subtract[dx<0] = 360
    direction = abs(subtract-direction)

    return direction
