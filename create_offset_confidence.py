#############################################################
# Created by Bodo Bookhagen and Ariane Mueting, August 2023 #
#############################################################

# Make sure to install a proper conda environment
# conda create -n CC python=3.11 numpy scipy pandas matplotlib ipython cupy tqdm numba -c conda-forge
# pip install install numba-progress

import glob, os, csv, sys, subprocess, tqdm, gzip, argparse
from datetime import datetime
from osgeo import gdal

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import correlation_confidence as cc

EXAMPLE = """There exist several approaches and philosophies to calculate uncertainties (or weights or confidences)
from image-correlation data. Here, we rely on a median approach for PlanetScope (both SD and PS2) data. The steps include:
1. Perform a 2D median filter with kernel_size=9 (variable) for each time step separately.
2. Take the median of all time steps to obtain an averaged value for each pixel.
3. Calculate the angle difference between this averaged value and each time step (x and y offset). Take the cosine of the angle difference to obtain a value between 0 and 1. All angle differences larger than 90 degree will be set to 0.
4. This will result in a pixel-based confidence value (or weight) - but the same value for x and y offsets.

A second method estimates uncertainties only for stable pixels. Here, offset direction sub-parallel to the hillslope angle

For estimating Landsat confidence, we use a different approach:


example for PlanetScope:
create_offset_confidence.py \
   --kernel_size 9 \
   --offset_tif_fn "disparity_maps/*_polyfit-F.tif" \
   --area_name aoi3 \
   --npy_out_path npy \
   --confidence_tif_out_path confidence \
   --sensor PS2

example for L8 (will require DEM for directional filtering):
create_offset_confidence.py \
   --kernel_size 7 \
   --offset_tif_fn L8/*-F.tif \
   --area_name aoi3 \
   --npy_out_path npy \
   --confidence_tif_out_path confidence \
   --sensor L8 \
   --dem_fname aoi6/output_COP30_matched_size.tif
"""

DESCRIPTION = """
Create confidence values (or weights or uncertainties) from dx and dy offset data. This can be used to weight a SBAS-like inversion for time series estimation.

Aug-2023, Bodo Bookhagen (bodo.bookhagen@uni-potsdam.de) and Ariane Mueting (mueting@uni-potsdam.de)
"""

def cmdLineParser():
    from argparse import RawTextHelpFormatter
    parser = argparse.ArgumentParser(description=DESCRIPTION, epilog=EXAMPLE, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--offset_tif_fn', help='2 Band offset file containing dx and dy data. Make sure to put into "quotes" when using wildcards (e.g., *).', required=True)
    parser.add_argument('--npy_out_path', help='Output compressed numpy files', required=True)
    parser.add_argument('--area_name', help='Name of area of interest', required=True)
    parser.add_argument('--confidence_tif_out_path', default='confidence', help='Output path for confidence files', required=False)
    parser.add_argument('-k', '--kernel_size', type=np.int8, default=9, help='Kernel Size for median filtering', required=False)
    parser.add_argument('--sensor',  default='PS', help='Sensor Name - L8, PS2, SD - for determining averaging method', required=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = cmdLineParser()

    kernel_size = args.kernel_size
    if os.path.exists(args.npy_out_path) == False:
        os.mkdir(args.npy_out_path)
    area_name = os.path.join(args.npy_out_path, args.area_name)

    #Setup filenames
    dx_npy_fname = area_name + "_dx.npy.gz"
    dx_median9_npy_fname = area_name + "_dx_median%02d.npy.gz"%kernel_size
    dx_median9_median_ar_npy_fname = area_name + "_dx_median%02d_median_ar.npy.gz"%kernel_size
    dx_std9_npy_fname = area_name + "_dx_std%02d.npy.gz"%kernel_size
    dx_median9_var_ar_npy_fname = area_name + "_dx_median%02d_var_ar.npy.gz"%kernel_size
    dy_npy_fname = area_name + "_dy.npy.gz"
    dy_median9_npy_fname = area_name + "_dy_median%02d.npy.gz"%kernel_size
    dy_std9_npy_fname = area_name + "_dy_std%02d.npy.gz"%kernel_size
    dy_median9_median_ar_npy_fname = area_name + "_dy_median%02d_median_ar.npy.gz"%kernel_size
    magnitude_stack_median_ar_npy_fname = area_name + "_magnitude_median_ar.npy.gz"
    # magnitude_stack_median_ar_5y_npy_fname = area_name + "_magnitude_median_ar_gt5y.npy.gz"
    # direction_stack_median_ar_5y_npy_fname = area_name + "_direction_median_ar_gt5y.npy.gz"
    direction_stack_median_ar_npy_fname = area_name + "_direction_median_ar.npy.gz"
    magnitude_stack_var_ar_npy_fname = area_name + "_magnitude_var_ar.npy.gz"
    direction_stack_var_ar_npy_fname = area_name + "_direction_var_ar.npy.gz"
    dy_median9_var_ar_npy_fname = area_name + "_dy_median%02d_var_ar.npy.gz"%kernel_size
    mask_npy_fname = area_name + "_mask.npy.gz"
    dx_mZscore_fname = area_name + "_dx_mZscore.npy.gz"
    dy_mZscore_fname = area_name + "_dy_mZscore.npy.gz"
    ts_dangle_npy_fname = area_name + "_ts_dangle.npy.gz"
    dem_dangle_npy_fname = area_name + "_dem_dangle.npy.gz"
    combined_score_npy_fname = area_name + "_combined_score.npy.gz"
    combined_score_median_ar_npy_fname = area_name + "_combined_score_median.npy.gz"
    combined_score_var_ar_npy_fname = area_name + "_combined_score_var.npy.gz"
    date0_stack_fname = area_name + "_date0.npy.gz"
    date1_stack_fname = area_name + "_date1.npy.gz"
    deltay_stack_fname = area_name + "_deltay.npy.gz"

    ### Convert TIF file and create numpy stack. Use mask and set pixels to NaN
    # load first dataset and get size of array
    filelist = glob.glob(args.offset_tif_fn)
    filelist.sort()
    #need an input tif to obtain projection information - use first file
    input_tif = filelist[0]
    ds = gdal.Open(input_tif)
    dxdy_size = ds.GetRasterBand(1).ReadAsArray().shape
    ds = None

    # load tif data and save as compressed npy
    cc.load_data(filelist, dxdy_size, output_path = args.npy_out_path, area_fname = args.area_name, mask=False, sensor='PS2')

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

    ### Load files: dx_stack
    print('Load dx data')
    f = gzip.GzipFile(dx_npy_fname, "r")
    dx_stack = np.load(f)
    f = None

    #f = gzip.GzipFile(mask_npy_fname, "r")
    #mask_stack = np.load(f)
    #f = None

    ### apply median filter (ignore nan and set all nan in dx_stack to nan)
    print('Calculate median filtered array with kernel_size = %d and ignoring nan'%kernel_size)
    if os.path.exists(dx_median9_npy_fname):
        print('\t Loading existing file')
        f = gzip.GzipFile(dx_median9_npy_fname, "r")
        dx_nanmedian9 = np.load(f)
        f.close()
        f = None
    else:
        # using same window size as correlation kernel
        dx_nanmedian9 = cc.filter2d_nanmedian_nomask(dx_stack, kernel_size=kernel_size)
        # save to file for later use
        f = gzip.GzipFile(dx_median9_npy_fname, "w")
        np.save(file=f, arr=dx_nanmedian9)
        f.close()
        f = None

    ### Calculate median of time series
    if os.path.exists(dx_median9_median_ar_npy_fname):
        print('\t Loading existing file')
        f = gzip.GzipFile(dx_median9_median_ar_npy_fname, "r")
        dx_stack_median_ar = np.load(f)
        f.close()
        f = None
    else:
        dx_stack_median_ar = cc.nanmedian_numba(dx_nanmedian9)
        if os.path.exists(dx_median9_median_ar_npy_fname) is False:
            f = gzip.GzipFile(dx_median9_median_ar_npy_fname, "w")
            np.save(file=f, arr=dx_stack_median_ar)
            f.close()
            f = None

    ### Calculate variance of time series
    if os.path.exists(dx_median9_var_ar_npy_fname):
        print('\t Loading existing file')
        f = gzip.GzipFile(dx_median9_var_ar_npy_fname, "r")
        dx_stack_var_ar = np.load(f)
        f.close()
        f = None
    else:
        dx_stack_var_ar = cc.nanvar_numba(dx_nanmedian9)
        if os.path.exists(dx_median9_var_ar_npy_fname) is False:
            f = gzip.GzipFile(dx_median9_var_ar_npy_fname, "w")
            np.save(file=f, arr=dx_stack_var_ar)
            f.close()
            f = None

    # not used for current approach: Z Score
    # print('Calculate modified Z score for x component for each pixel and timestep')
    # if os.path.exists(dx_mZscore_fname):
    #     print('\t Loading existing file')
    #     f = gzip.GzipFile(dx_mZscore_fname, "r")
    #     dx_mZscore = np.load(f)
    #     f.close()
    #     f = None
    # else:
    #     dx_mZscore = cc.mZscore_numba(dx_stack, dx_stack_median_ar)
    #     #remove inf values from division by zero (only numba version
    #     dx_mZscore[np.isinf(dx_mZscore)] = np.float32(np.nan)
    #     # normalize modified zScore with 2nd and 98th percentile of all data and then clip
    #     p02, p98 = np.nanpercentile(dx_mZscore, [2, 98])
    #     dx_mZscore = (dx_mZscore - p02)  / (p98 - p02)
    #     dx_mZscore = np.clip(dx_mZscore, 0, 1)
    #     if os.path.exists(dx_mZscore_fname) is False:
    #         print('\t saving to gzipped npy files')
    #         f = gzip.GzipFile(dx_mZscore_fname, "w")
    #         np.save(file=f, arr=dx_mZscore)
    #         f.close()
    #         f = None

    ### Repeat for dy_stack
    print('Load dy data')
    f = gzip.GzipFile(dy_npy_fname, "r")
    dy_stack = np.load(f)
    f = None
    print('Calculate median filtered array with kernel_size = %d and ignoring nan'%kernel_size)
    if os.path.exists(dy_median9_npy_fname):
        print('\t Loading existing file')
        f = gzip.GzipFile(dy_median9_npy_fname, "r")
        dy_nanmedian9 = np.load(f)
        f.close()
        f = None
    else:
        # using same window size as correlation kernel
        dy_nanmedian9 = cc.filter2d_nanmedian_nomask(dy_stack, kernel_size=kernel_size)
        # save to file for later use
        f = gzip.GzipFile(dy_median9_npy_fname, "w")
        np.save(file=f, arr=dy_nanmedian9)
        f.close()
        f = None


    ### Calculate median of time series
    if os.path.exists(dy_median9_median_ar_npy_fname):
        print('\t Loading existing file')
        f = gzip.GzipFile(dy_median9_median_ar_npy_fname, "r")
        dy_stack_median_ar = np.load(f)
        f.close()
        f = None
    else:
        dy_stack_median_ar = cc.nanmedian_numba(dy_nanmedian9)
        if os.path.exists(dy_median9_median_ar_npy_fname) is False:
            f = gzip.GzipFile(dy_median9_median_ar_npy_fname, "w")
            np.save(file=f, arr=dy_stack_median_ar)
            f.close()
            f = None

    ### Calculate variance of time series
    if os.path.exists(dy_median9_var_ar_npy_fname):
        print('\t Loading existing file')
        f = gzip.GzipFile(dy_median9_var_ar_npy_fname, "r")
        dy_stack_var_ar = np.load(f)
        f.close()
        f = None
    else:
        dy_stack_var_ar = cc.nanvar_numba(dy_nanmedian9)
        if os.path.exists(dy_median9_var_ar_npy_fname) is False:
            f = gzip.GzipFile(dy_median9_var_ar_npy_fname, "w")
            np.save(file=f, arr=dy_stack_var_ar)
            f.close()
            f = None

    # not used for current approach
    # print('Calculate modified Z score for y component for each pixel and timestep')
    # if os.path.exists(dy_mZscore_fname):
    #     print('\t Loading existing file')
    #     f = gzip.GzipFile(dy_mZscore_fname, "r")
    #     dy_mZscore = np.load(f)
    #     f.close()
    #     f = None
    # else:
    #     dy_mZscore = cc.mZscore_numba(dy_stack, dy_stack_median_ar)
    #     #remove inf values from division by zero (only numba version
    #     dy_mZscore[np.isinf(dy_mZscore)] = np.float32(np.nan)
    #     # normalize modified zScore with 2nd and 98th percentile of all data and then clip
    #     p02, p98 = np.nanpercentile(dy_mZscore, [2, 98])
    #     dy_mZscore = (dy_mZscore - p02)  / (p98 - p02)
    #     dy_mZscore = np.clip(dy_mZscore, 0, 1)
    #     if os.path.exists(dy_mZscore_fname) is False:
    #         print('\t saving to gzipped npy files')
    #         f = gzip.GzipFile(dy_mZscore_fname, "w")
    #         np.save(file=f, arr=dy_mZscore)
    #         f.close()
    #         f = None

    print('Calculate angle difference between median-filtered average and pixel values for each timestep')
    if os.path.exists(ts_dangle_npy_fname):
        print('\t Loading existing file')
        f = gzip.GzipFile(ts_dangle_npy_fname, "r")
        ts_dangle = np.load(f)
        f.close()
        f = None
    else:
        ts_dangle, ts_dangle_median, ts_dangle_var = cc.calc_dangle(dx_stack, dy_stack,
            dx_stack_median_ar, dy_stack_median_ar)
        if os.path.exists(ts_dangle_npy_fname) is False:
            f = gzip.GzipFile(ts_dangle_npy_fname, "w")
            np.save(file=f, arr=ts_dangle)
            f.close()
            f = None

    # here we use only the median angle difference for Planet

    if args.sensor == 'PS2' or args.sensor == 'SD':
        combined_score = ts_dangle
    elif args.sensor == 'L8':
        ### USE Dem for direction filtering
        dem_fname = args.dem_fname
        aspect_out_fname = dem_fname[-4] + '_aspect%d.tif'%kernel_size
        slope_out_fname = dem_fname[-4] + '_slope%d.tif'%kernel_size
        dem_slope, dem_aspect = cc.aspect_slope_dem(dem_fname, aspect_out_fname, slope_out_fname, kernel_size = kernel_size)

        print('Calculate angle difference between dem aspect and time series pixel values for each timestep')
        if os.path.exists(dem_dangle_npy_fname):
            print('\t Loading existing file')
            f = gzip.GzipFile(dem_dangle_npy_fname, "r")
            dem_dangle = np.load(f)
            f.close()
            f = None
        else:
            dem_dangle, dem_dangle_median_ts, dem_dangle_var_ts = cc.calc_dangle_dem(dx_stack, dy_stack,
                dem_aspect)
            if os.path.exists(dem_dangle_npy_fname) is False:
                f = gzip.GzipFile(dem_dangle_npy_fname, "w")
                np.save(file=f, arr=dem_dangle)
                f.close()
                f = None

    # ### combine the three metrics into one score
    # combined_score = ts_dangle*dx_mZscore*dy_mZscore
    if os.path.exists(combined_score_npy_fname) is False:
        f = gzip.GzipFile(combined_score_npy_fname, "w")
        np.save(file=f, arr=combined_score)
        f.close()
        f = None

    ### calculate median combined score
    if os.path.exists(combined_score_median_ar_npy_fname):
        print('\t Loading existing file')
        f = gzip.GzipFile(combined_score_median_ar_npy_fname, "r")
        combined_score_median_ar = np.load(f)
        f.close()
        f = None
    else:
        combined_score_median_ar = cc.nanmedian_numba(combined_score)
        if os.path.exists(combined_score_median_ar_npy_fname) is False:
            f = gzip.GzipFile(combined_score_median_ar_npy_fname, "w")
            np.save(file=f, arr=combined_score_median_ar)
            f.close()
            f = None

    ### calculate variance of combined score
    if os.path.exists(combined_score_var_ar_npy_fname):
        print('\t Loading existing file')
        f = gzip.GzipFile(combined_score_var_ar_npy_fname, "r")
        combined_score_var_ar = np.load(f)
        f.close()
        f = None
    else:
        combined_score_var_ar = cc.nanvar_numba(combined_score)
        if os.path.exists(combined_score_var_ar_npy_fname) is False:
            f = gzip.GzipFile(combined_score_var_ar_npy_fname, "w")
            np.save(file=f, arr=combined_score_var_ar)
            f.close()
            f = None

    ### Calculate Velocity direction (median)
    if os.path.exists(direction_stack_median_ar_npy_fname):
        print('\t Loading existing file')
        f = gzip.GzipFile(direction_stack_median_ar_npy_fname, "r")
        direction_stack_median = np.load(f)
        f.close()
        f = None
    else:
        direction_stack_median = np.rad2deg(np.arctan2(dy_stack_median_ar, dx_stack_median_ar))
        if os.path.exists(direction_stack_median_ar_npy_fname) is False:
            f = gzip.GzipFile(direction_stack_median_ar_npy_fname, "w")
            np.save(file=f, arr=direction_stack_median)
            f.close()
            f = None

    ### Calculate Velocity magnitude (median)
    if os.path.exists(magnitude_stack_median_ar_npy_fname):
        print('\t Loading existing file')
        f = gzip.GzipFile(magnitude_stack_median_ar_npy_fname, "r")
        magnitude_stack_median = np.load(f)
        f.close()
        f = None
    else:
        magnitude_stack_median = np.sqrt(dx_stack_median_ar**2 + dy_stack_median_ar**2)
        if os.path.exists(magnitude_stack_median_ar_npy_fname) is False:
            f = gzip.GzipFile(magnitude_stack_median_ar_npy_fname, "w")
            np.save(file=f, arr=magnitude_stack_median)
            f.close()
            f = None

    ### Calculate Velocity direction (variance)
    if os.path.exists(direction_stack_var_ar_npy_fname):
        print('\t Loading existing file')
        f = gzip.GzipFile(direction_stack_var_ar_npy_fname, "r")
        direction_stack_var = np.load(f)
        f.close()
        f = None
    else:
        direction_stack_var = np.rad2deg(np.arctan2(dy_stack_var_ar, dx_stack_var_ar))
        if os.path.exists(direction_stack_var_ar_npy_fname) is False:
            f = gzip.GzipFile(direction_stack_var_ar_npy_fname, "w")
            np.save(file=f, arr=direction_stack_var)
            f.close()
            f = None

    ### Calculate Velocity magnitude (variance)
    if os.path.exists(magnitude_stack_var_ar_npy_fname):
        print('\t Loading existing file')
        f = gzip.GzipFile(magnitude_stack_var_ar_npy_fname, "r")
        magnitude_stack_var = np.load(f)
        f.close()
        f = None
    else:
        magnitude_stack_var = np.sqrt(dx_stack_var_ar**2 + dy_stack_var_ar**2)
        if os.path.exists(magnitude_stack_var_ar_npy_fname) is False:
            f = gzip.GzipFile(magnitude_stack_var_ar_npy_fname, "w")
            np.save(file=f, arr=magnitude_stack_var)
            f.close()
            f = None


    ### Plot figures
    stack_median_var_4plots_fname = area_name + '_dxdy_median_var_4plots.png'
    stack_median_direction_magntitude_4plots_fname = area_name + '_median_direction_magnitude_4plots.png'
    combined_scores_min_max_fname = area_name + '_combined_scores_8plots.png'

    cc.plot_dxdy_median(dx_stack_median_ar, dy_stack_median_ar, dx_stack_var_ar, dy_stack_var_ar,
                        date0_stack.shape[0], stack_median_var_4plots_fname)
    cc.plot_direction_magnitude(direction_stack_median, magnitude_stack_median, direction_stack_var,
                        magnitude_stack_var, date0_stack.shape[0], stack_median_direction_magntitude_4plots_fname)


    combined_score_median_ts = cc.nanmedian_numba_ts(combined_score)
    imax = np.where(np.nanmax(combined_score_median_ts) == combined_score_median_ts)[0][0]
    imin = np.where(np.nanmin(combined_score_median_ts) == combined_score_median_ts)[0][0]
    cc.plot_2example_2metrics(imin, imax, ts_dangle,
            combined_score, date0_stack, date1_stack, deltay_stack, combined_scores_min_max_fname)


    # Write to Geotiff files
    cc.write_Geotiff(input_tif, magnitude_stack_median, area_name + '_magnitude_stack_median.tif')
    cc.write_Geotiff(input_tif, direction_stack_median, area_name + '_direction_stack_median.tif')
    cc.write_Geotiff(input_tif, combined_score_median_ar, area_name + '_combined_confidence_median.tif')

    # Export Angle difference to tif files (each time step)
    if os.path.exists(args.confidence_tif_out_path) == False:
        os.mkdir(args.confidence_tif_out_path)
    if args.sensor == 'PS2' or args.sensor == 'SD':
        cc.write_Geotiff_ts(input_tif, ts_dangle, date0_stack, date1_stack,
            output_prefix=args.area_name, output_postfix='confidence', output_dir=args.confidence_tif_out_path)

    if args.sensor == 'L8':
        # Export Angle difference to tif files (each time step)
        cc.write_Geotiff_ts(input_tif, dem_dangle, date0_stack, date1_stack,
            output_prefix=args.area_name, output_postfix='confidence', output_dir=args.confidence_tif_out_path)

        cc.write_Geotiff(input_tif, dem_dangle_median_ar, area_name + '_dem_dangle_median_confidence.tif')
        cc.write_Geotiff(input_tif, dem_dangle_var_ar, area_name + 'dem_dangle_var_confidence.tif')

        # ### calculate
        # dem_dangle_median_ar = cc.nanmedian_numba(dem_dangle)
        # dem_dangle_var_ar = cc.nanvar_numba(dem_dangle)
        # dem_dangle_median_ts = cc.nanmedian_numba_ts(dem_dangle)
        # imax = np.where(np.nanmax(dem_dangle_median_ts) == dem_dangle_median_ts)[0][0]
        # imin = np.where(np.nanmin(dem_dangle_median_ts) == dem_dangle_median_ts)[0][0]
        # combined_scores_min_max_fname = area_name + '_combined_scores_dem_8plots.png'
        # combined_score = dem_dangle * dx_mZscore * dy_mZscore
        # cc.plot_2example_metrics(imin, imax, dem_dangle, dx_mZscore, dy_mZscore,
        #         combined_score, date0_stack, date1_stack, deltay_stack, combined_scores_min_max_fname)


    # Alternatively: calculate standard deviation in x and y direction and use this as confidence value
    # not used in this context
    # # Calculate standard deviation for dx and dy
    # print('Calculate std. dev. of dx for every pixel for each timestep')
    # if os.path.exists(dx_std9_npy_fname):
    #     print('\t Loading existing file')
    #     f = gzip.GzipFile(dx_std9_npy_fname, "r")
    #     dx_stack_std = np.load(f)
    #     f.close()
    #     f = None
    # else:
    #     dx_stack_std = cc.filter2d_nanstddev(dx_stack, mask_stack, kernel_size=9)
    #     if os.path.exists(dx_std9_npy_fname) is False:
    #         f = gzip.GzipFile(dx_std9_npy_fname, "w")
    #         np.save(file=f, arr=dx_stack_std)
    #         f.close()
    #         f = None
    #
    # input_tif = '/raid/L8_DelMedio/disparity_maps/20140922_20210826_DelMedio_mgm_ck9-F.tif'
    # cc.write_Geotiff_ts(input_tif, dx_stack_std, date0_stack, date1_stack,
    #     output_prefix='DelMedio_mgm_ck9-F_dx_std', output_dir='/raid/L8_DelMedio/disparity_maps')
    #
    # print('Calculate std. dev. of dy for every pixel for each timestep')
    # if os.path.exists(dy_std9_npy_fname):
    #     print('\t Loading existing file')
    #     f = gzip.GzipFile(dy_std9_npy_fname, "r")
    #     dy_stack_std = np.load(f)
    #     f.close()
    #     f = None
    # else:
    #     dy_stack_std = cc.filter2d_nanstddev(dy_stack, mask_stack, kernel_size=9)
    #     if os.path.exists(dy_std9_npy_fname) is False:
    #         f = gzip.GzipFile(dy_std9_npy_fname, "w")
    #         np.save(file=f, arr=dy_stack_std)
    #         f.close()
    #         f = None
    #
    # input_tif = '/raid/L8_DelMedio/disparity_maps/20140922_20210826_DelMedio_mgm_ck9-F.tif'
    # cc.write_Geotiff_ts(input_tif, dy_stack_std, date0_stack, date1_stack,
    #     output_prefix='DelMedio_mgm_ck9-F_dy_std', output_dir='/raid/L8_DelMedio/disparity_maps')
