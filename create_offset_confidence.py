#############################################################
# Created by Bodo Bookhagen and Ariane Mueting, August 2023 #
#############################################################

import glob, os, csv, sys, subprocess, tqdm, gzip, argparse
from datetime import datetime
from osgeo import gdal

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skimage import measure
import correlation_confidence as cc

#TODO: sensor is not needed, remove
#TODO: 2 or 3 Band offset file? check if 3 band also works
EXAMPLE = """

example for PlanetScope:
create_offset_confidence.py \
    --method 1 \
    --kernel_size 9 \
    --threshold_angle 45 \
    --threshold_size 1000 \
    --offset_tif_fn "disparity_maps/*_polyfit-F.tif" \
    --area_name aoi3 \
    --npy_out_path npy \
    --confidence_tif_out_path confidence \
    --sensor PS

create_offset_confidence.py \
    --method 2 \
    --kernel_size 9 \
    --threshold_angle 45 \
    --threshold_size 1000 \
    --offset_tif_fn "disparity_maps/*_polyfit-F.tif" \
    --area_name aoi3 \
    --npy_out_path npy \
    --confidence_tif_out_path confidence \
    --sensor PS


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
Create confidence values (or weights or uncertainties) from dx and dy offset data. This can be used to create confidence values (weights) for a SBAS-like inversion for time series estimation.

There exist several approaches and philosophies to calculate uncertainties (or weights or confidences)
from image-correlation data. Here, we have implemented the following methods:

Method 1 (Useful for PlanetScope or other well-behaved datasets):
1. Perform a 2D median filter with kernel_size=9 (variable) for each time step separately. This smoothes each timestep.
2. Take the median of all time steps to obtain an averaged value for each pixel.
3. Calculate the angle difference between this averaged value and each time step (x and y offset). Take the cosine of the angle difference to obtain a value between 0 and 1. All angle differences larger than 90 degree will be set to 0.
4. This will result in a pixel-based confidence value (or weight) - but the same value for x and y offsets.
5. Output a mask file with pixels for each time step that have been flagged as: nan by the correation and 0 confidence value (angle difference above 90 degree). This is different for each time step.


Method 2 (Useful for PlanetScope or other well-behaved datasets):
1. Calculate the std. deviation of all directions through time.
2. If std. deviation is larger than 45 degree (or a threshold angle), the terrain is not considered a landslide and masked out. This is useful for creating a mask of stable terrain.
3. For the unstable terrain (i.e., landslides), calculate the angle difference between time-averaged averaged value and value at each time step. Take the cosine of the angle difference to obtain a value between 0 and 1. All angle differences larger than 90 degree will be set to 0.
4. Create a mask that shows all unstable terrain (same mask for all time steps).


Method 3 (Useful for noisy offsets such as derived from Landsat and Sentinel 2):
1. Perform a 2D median filter with kernel_size=9 (variable) for each time step separately. This smoothes each timestep.
2. Calculate angle difference between aspect direction derived from a DEM and offset direction. If larger than 45 degree (or a threshold value), the pixel and timestep receives 0 weight. Otherwise, the cosine of 2 * the angle difference is assigned as weight.

Aug-2023, Bodo Bookhagen (bodo.bookhagen@uni-potsdam.de) and Ariane Mueting (mueting@uni-potsdam.de)
"""

def cmdLineParser():
    from argparse import RawTextHelpFormatter
    parser = argparse.ArgumentParser(description=DESCRIPTION, epilog=EXAMPLE, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--method', default=1, type=np.int8, help='Confidence-value method to chose from. See Description.', required=True)
    parser.add_argument('--offset_tif_fn', help='2 Band offset file containing dx and dy data. Make sure to put into "quotes" when using wildcards (e.g., *).', required=True)
    if sys.version_info[1] < 9:
        parser.add_argument('--mask', type=bool, default=False, help='Set to True if you have three-band TIF files and the third band contains the mask band (default=False).', required=False)
    else:
        parser.add_argument('--mask', type=argparse.BooleanOptionalAction, default=False, help='Set to True if you have three-band TIF files and the third band contains the mask band.', required=False)
    parser.add_argument('--npy_out_path', default='npy', help='Output compressed numpy files', required=True)
    parser.add_argument('--threshold_angle', default=45, type=np.int8, help='Threshold angle in degree for filtering out angle deivations', required=False)
    parser.add_argument('--area_name', help='Name of area of interest', required=True)
    parser.add_argument('--threshold_size', default=0, type=np.int16, help='Threshold size in pixels to remove from mask image. Only continuous patches above this size are kept. Set to 10 or larger for useful results.', required=False)
    parser.add_argument('--confidence_tif_out_path', default='confidence', help='Output path for confidence files', required=False)
    parser.add_argument('-k', '--kernel_size', type=np.int8, default=9, help='Kernel Size for median filtering', required=False)
    parser.add_argument('--sensor',  default='PS', help='Sensor Name - L8 or PS - for determining averaging method', required=False, choices =["PS", "L8"])
    return parser.parse_args()


if __name__ == '__main__':
    args = cmdLineParser()

    # Debugging
#    parser = argparse.ArgumentParser(description='Create confidence values')
#    args = parser.parse_args()
#    args.method=2
#    args.offset_tif_fn="disparity_maps/*_polyfit-F.tif"
#    args.mask = False
#    args.npy_out_path='npy'
#    args.area_name='aoi3'
#    args.confidence_tif_out_path='confidence'
#    args.kernel_size=9
#    args.sensor='PS'
#    args.threshold_size = 1000


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
    ts_dangle_mask_npy_fname = area_name + "_ts_dangle_mask.npy.gz"
    dem_dangle_npy_fname = area_name + "_dem_dangle.npy.gz"
    combined_score_npy_fname = area_name + "_combined_score.npy.gz"
    combined_score_median_ar_npy_fname = area_name + "_combined_score_median.npy.gz"
    combined_score_var_ar_npy_fname = area_name + "_combined_score_var.npy.gz"
    date0_stack_fname = area_name + "_date0.npy.gz"
    date1_stack_fname = area_name + "_date1.npy.gz"
    deltay_stack_fname = area_name + "_deltay.npy.gz"
    ts_dangle_mask_npy_fname = area_name + '_filtered_ts_dangle_mask.npy.gz'
    directions_sd_mask_npy_fname = area_name + '_directions_sd_mask.npy.gz'
    directions_sd_mask_geotiff_fname = area_name + '_directions_sd_mask.tif'

    # load first dataset and get size of array
    filelist = glob.glob(args.offset_tif_fn)
    filelist.sort()
    #need an input tif to obtain projection information - use first file
    print('Open one tif to obtain geodata information')
    input_tif = filelist[0]
    ds = gdal.Open(input_tif)
    dxdy_size = ds.GetRasterBand(1).ReadAsArray().shape
    ds = None

    ### Convert TIF file and create numpy stack. Use mask and set pixels to NaN
    # load tif data and save as compressed npy
    cc.load_data(filelist, dxdy_size, output_path = args.npy_out_path, area_fname = args.area_name, mask=False, sensor=args.sensor)

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

    if args.method == 1:
        # Method 1 (Useful for PlanetScope or other well-behaved datasets):
        # 1. Perform a 2D median filter with kernel_size=9 (variable) for each time step separately. This smoothes each timestep.
        # 2. Take the median of all time steps to obtain an averaged value for each pixel.
        # 3. Calculate the angle difference between this averaged value and each time step (x and y offset). Take the cosine of the angle difference to obtain a value between 0 and 1. All angle differences larger than 90 degree will be set to 0.
        # 4. This will result in a pixel-based confidence value (or weight) - but the same value for x and y offsets.
        # 5. Output a mask file with pixels for each time step that have been flagged as: nan by the correation and 0 confidence value (angle difference above 90 degree). This is different for each time step.
        # Mask file contains 0 for nan and 1 for data regions

        print('Load mask data')
        f = gzip.GzipFile(mask_npy_fname, "r")
        mask_stack = np.load(f)
        f = None

        ### apply median filter (ignore nan and set all nan in dx_stack to nan)
        print('Calculate median filtered array with kernel_size = %d and ignoring nan'%kernel_size)
        if os.path.exists(dx_median9_npy_fname):
            print('\t Loading existing file')
            f = gzip.GzipFile(dx_median9_npy_fname, "r")
            dx_nanmedian9 = np.load(f)
            f.close()
            f = None
        else:
            ### Load files: dx_stack
            print('Load dx data')
            f = gzip.GzipFile(dx_npy_fname, "r")
            dx_stack = np.load(f)
            f = None
            # using same window size as correlation kernel
            # dx_nanmedian9 = cc.filter2d_nanmedian_nomask(dx_stack, kernel_size=kernel_size)
            dx_nanmedian9 = cc.filter2d_nanmedian(dx_stack, mask_stack, kernel_size=kernel_size)
            # save to file for later use
            f = gzip.GzipFile(dx_median9_npy_fname, "w")
            np.save(file=f, arr=dx_nanmedian9)
            f.close()
            f = None
            dx_stack = None

        ### Calculate median of median-filtered time series
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
        print('Calculate median filtered array with kernel_size = %d and ignoring nan'%kernel_size)
        if os.path.exists(dy_median9_npy_fname):
            print('\t Loading existing file')
            f = gzip.GzipFile(dy_median9_npy_fname, "r")
            dy_nanmedian9 = np.load(f)
            f.close()
            f = None
        else:
            print('Load dy data')
            f = gzip.GzipFile(dy_npy_fname, "r")
            dy_stack = np.load(f)
            f = None
            # using same window size as correlation kernel
            # dy_nanmedian9 = cc.filter2d_nanmedian_nomask(dy_stack, kernel_size=kernel_size)
            dy_nanmedian9 = cc.filter2d_nanmedian(dy_stack, mask_stack, kernel_size=kernel_size)
            # save to file for later use
            f = gzip.GzipFile(dy_median9_npy_fname, "w")
            np.save(file=f, arr=dy_nanmedian9)
            f.close()
            f = None
            dy_stack = None

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
        if os.path.exists(ts_dangle_npy_fname) and os.path.exists(ts_dangle_mask_npy_fname):
            print('\t Loading existing file')
            f = gzip.GzipFile(ts_dangle_npy_fname, "r")
            ts_dangle = np.load(f)
            f.close()
            f = None
        else:
            # here we are running angle difference calculation with the smoothed median values - could also be done with the original dx and dy values
            # ts_dangle, ts_dangle_median, ts_dangle_var, ts_dangle_mask = cc.calc_dangle(dx_nanmedian9, dy_nanmedian9,
            #     dx_stack_median_ar, dy_stack_median_ar, mask_stack)
            ts_dangle, ts_dangle_median, ts_dangle_var, ts_dangle_mask = cc.calc_dangle(dx_nanmedian9, dy_nanmedian9,
                dx_stack_median_ar, dy_stack_median_ar, mask_stack)
            if os.path.exists(ts_dangle_npy_fname) is False:
                f = gzip.GzipFile(ts_dangle_npy_fname, "w")
                np.save(file=f, arr=ts_dangle)
                f.close()
                f = None
            if os.path.exists(ts_dangle_mask_npy_fname) is False:
                f = gzip.GzipFile(ts_dangle_mask_npy_fname, "w")
                np.save(file=f, arr=ts_dangle_mask)
                f.close()
                f = None


        if args.threshold_size > 0:
            if os.path.exists(ts_dangle_mask_npy_fname):
                print('\t Loading existing file')
                f = gzip.GzipFile(ts_dangle_mask_npy_fname, "r")
                ts_dangle_mask = np.load(f)
                f.close()
                f = None
            else:
                #filter ts_dangle_mask by size of patches - keep only continues patches with more than args.threshold_size
                print('Filter masked image with patch size: %d pixels'%args.threshold_size)
                for i in tqdm.tqdm(range(ts_dangle_mask.shape[0])):
                    # iterate through all time steps - this is still slow
                    dbin = ts_dangle_mask[i,:,:]
                    labeled = measure.label(dbin, background=0, connectivity=2)
                    info = measure.regionprops(labeled)
                    # Filter connected components based on size
                    filtered_labels = []
                    for region in info:
                        if region.area > args.threshold_size:
                            filtered_labels.append(region.label)

                    filtered_mask = np.isin(labeled, filtered_labels)
                    ts_dangle_mask[i,:,:] = filtered_mask # write filtered results back to original array

                if os.path.exists(ts_dangle_mask_npy_fname) is False:
                    f = gzip.GzipFile(ts_dangle_mask_npy_fname, "w")
                    np.save(file=f, arr=ts_dangle_mask)
                    f.close()
                    f = None


        # ### combine the metrics into one score
        combined_score = ts_dangle
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
            direction_stack_median[np.abs(direction_stack_median) >= 90] = np.nan
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
            magnitude_stack_median[np.isnan(direction_stack_median)] = np.nan
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
            # direction_stack_var[np.abs(direction_stack_var) > 45] = np.nan
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
        combined_scores_min_max_fname = area_name + '_combined_scores_6plots.png'
        masksum_fname = area_name + '_mask_sum.png'

        cc.plot_dxdy_median(dx_stack_median_ar, dy_stack_median_ar, dx_stack_var_ar, dy_stack_var_ar,
                            date0_stack.shape[0], stack_median_var_4plots_fname)
        cc.plot_direction_magnitude(direction_stack_median, magnitude_stack_median, direction_stack_var,
                            magnitude_stack_var, date0_stack.shape[0], stack_median_direction_magntitude_4plots_fname)

        # create sum of nan pixels
        ts_dangle_mask_sum = np.sum(ts_dangle_mask, axis=0)
        cc.plot_mask_sum(ts_dangle_mask_sum, date0_stack.shape[0], masksum_fname)

        combined_score_median_ts = cc.nanmedian_numba_ts(combined_score)
        imax = np.where(np.nanmax(combined_score_median_ts) == combined_score_median_ts)[0][0]
        imin = np.where(np.nanmin(combined_score_median_ts) == combined_score_median_ts)[0][0]
        cc.plot_2example_3metrics(imin, imax, ts_dangle,
                combined_score, ts_dangle_mask, date0_stack, date1_stack, deltay_stack, combined_scores_min_max_fname)

        # Write to Geotiff files
        cc.write_Geotiff(input_tif, magnitude_stack_median, area_name + '_magnitude_stack_median.tif')
        cc.write_Geotiff(input_tif, direction_stack_median, area_name + '_direction_stack_median.tif')
        cc.write_Geotiff(input_tif, combined_score_median_ar, area_name + '_combined_confidence_median.tif')
        cc.write_Geotiff(input_tif, ts_dangle_mask_sum, area_name + '_mask_sum.tif')

        # Export Angle difference to tif files (each time step)
        if os.path.exists(args.confidence_tif_out_path) == False:
            os.mkdir(args.confidence_tif_out_path)
        cc.write_Geotiff_ts(input_tif, ts_dangle, date0_stack, date1_stack,
                output_prefix=args.area_name, output_postfix='confidence', output_dir=args.confidence_tif_out_path)
        cc.write_Geotiff_ts(input_tif, ts_dangle_mask, date0_stack, date1_stack,
                                            output_prefix=args.area_name, output_postfix='mask', output_dir=args.confidence_tif_out_path)


    if args.method == 2:
        # 1. Calculate the std. deviation of all directions through time.
        # 2. If std. deviation is larger than 45 degree (or a threshold angle), the terrain is not considered a landslide and masked out. This is useful for creating a mask of stable terrain.
        # 3. For the unstable terrain (i.e., landslides), calculate the angle difference between time-averaged averaged value and value at each time step. Take the cosine of the angle difference to obtain a value between 0 and 1. All angle differences larger than 90 degree will be set to 0.
        # 4. Create a mask that shows all unstable terrain (same mask for all time steps).
        print('Load mask data')
        f = gzip.GzipFile(mask_npy_fname, "r")
        mask_stack = np.load(f)
        f = None

        ### Load files: dx_stack and dy_stack
        print('Load dx data')
        f = gzip.GzipFile(dx_npy_fname, "r")
        dx_stack = np.load(f)
        f = None

        print('Load dy data')
        f = gzip.GzipFile(dy_npy_fname, "r")
        dy_stack = np.load(f)
        f = None

        # could also use median-smoothed dx and dy stack
        directions = cc.calc_angle_numba(dx_stack, dy_stack) # returns angles in degree
        del dx_stack
        del dy_stack # remove from memory
        print('Calculating std. dev. of angles through time')
        # dir_var = cc.angle_variance(directions) # angle_variance scaled between 0 and 1
        directions_sd = cc.nanstd_numba(directions)
        mask = np.where(directions_sd < args.threshold_angle, 1, 0) # use this mask

        if args.threshold_size > 0:
            #filter mask by size of patches - keep only continues patches with more than args.threshold_size
            print('Filter masked image with patch size: %d pixels'%args.threshold_size)
            # iterate through all time steps - this is still slow
            labeled = measure.label(mask, background=0, connectivity=2)
            info = measure.regionprops(labeled)
            # Filter connected components based on size
            filtered_labels = []
            for region in info:
                if region.area > args.threshold_size:
                    filtered_labels.append(region.label)

            filtered_mask = np.isin(labeled, filtered_labels)
            mask = filtered_mask # write results back into original array

        # Plot directions and mask
        directions_sd_fname = area_name + '_directions_sd.png'
        cc.plot_direction_sd_mask(directions_sd, mask, date0_stack.shape[0], directions_sd_fname)

        # write directions to geotiff
        cc.write_Geotiff(input_tif, mask, directions_sd_mask_geotiff_fname)

        # write filtered mask to numpy array
        if os.path.exists(directions_sd_mask_npy_fname) is False:
            f = gzip.GzipFile(directions_sd_mask_npy_fname, "w")
            np.save(file=f, arr=mask)
            f.close()
            f = None

        # Use mask (only valid areas) and calculate confidence from direction offset only for landslide areas


        # Export mask to tif files (mask is the same for each time step)
        if os.path.exists(args.confidence_tif_out_path) == False:
            os.mkdir(args.confidence_tif_out_path)
        # cc.write_Geotiff_ts(input_tif, ts_dangle, date0_stack, date1_stack,
        #        output_prefix=args.area_name, output_postfix='confidence', output_dir=args.confidence_tif_out_path)
        cc.write_Geotiff_ts_mask(input_tif, mask, date0_stack, date1_stack,
                                            output_prefix=args.area_name, output_postfix='mask', output_dir=args.confidence_tif_out_path)


    if args.method == 3:
        #Angle difference between DEM Aspect and direction
        ### Use Dem for direction filtering
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
            ### apply median filter (ignore nan and set all nan in dx_stack to nan)
            if os.path.exists(dx_median9_npy_fname):
                print('\t Loading existing file')
                f = gzip.GzipFile(dx_median9_npy_fname, "r")
                dx_nanmedian9 = np.load(f)
                f.close()
                f = None
            else:
                ### Load files: dx_stack
                print('Load dx data')
                f = gzip.GzipFile(dx_npy_fname, "r")
                dx_stack = np.load(f)
                f = None

                print('Load mask data')
                f = gzip.GzipFile(mask_npy_fname, "r")
                mask_stack = np.load(f)
                f = None

                print('Calculate median filtered array with kernel_size = %d and ignoring nan'%kernel_size)
                # using same window size as correlation kernel
                # dx_nanmedian9 = cc.filter2d_nanmedian_nomask(dx_stack, kernel_size=kernel_size)
                dx_nanmedian9 = cc.filter2d_nanmedian(dx_stack, mask_stack, kernel_size=kernel_size)
                # save to file for later use
                f = gzip.GzipFile(dx_median9_npy_fname, "w")
                np.save(file=f, arr=dx_nanmedian9)
                f.close()
                f = None
                dx_stack = None
                mask_stack = None

            ### Repeat for dy_stack
            if os.path.exists(dy_median9_npy_fname):
                print('\t Loading existing file')
                f = gzip.GzipFile(dy_median9_npy_fname, "r")
                dy_nanmedian9 = np.load(f)
                f.close()
                f = None
            else:
                print('Load dy data')
                f = gzip.GzipFile(dy_npy_fname, "r")
                dy_stack = np.load(f)
                f = None

                print('Load mask data')
                f = gzip.GzipFile(mask_npy_fname, "r")
                mask_stack = np.load(f)
                f = None

                print('Calculate median filtered array with kernel_size = %d and ignoring nan'%kernel_size)
                # using same window size as correlation kernel
                # dy_nanmedian9 = cc.filter2d_nanmedian_nomask(dy_stack, kernel_size=kernel_size)
                dy_nanmedian9 = cc.filter2d_nanmedian(dy_stack, mask_stack, kernel_size=kernel_size)
                # save to file for later use
                f = gzip.GzipFile(dy_median9_npy_fname, "w")
                np.save(file=f, arr=dy_nanmedian9)
                f.close()
                f = None
                dy_stack = None
                mask_stack = None

            # Using median-filtered dx and dy values - could also use original values (not smoothed)
            dem_dangle, dem_dangle_median_ts, dem_dangle_var_ts = cc.calc_dangle_dem(dx_nanmedian9, dy_nanmedian9,
                dem_aspect)
            if os.path.exists(dem_dangle_npy_fname) is False:
                f = gzip.GzipFile(dem_dangle_npy_fname, "w")
                np.save(file=f, arr=dem_dangle)
                f.close()
                f = None

        # Export Angle difference to tif files (each time step)
        cc.write_Geotiff_ts(input_tif, dem_dangle, date0_stack, date1_stack,
            output_prefix=args.area_name, output_postfix='confidence', output_dir=args.confidence_tif_out_path)

        cc.write_Geotiff(input_tif, dem_dangle_median_ar, area_name + '_dem_dangle_median_confidence.tif')
        cc.write_Geotiff(input_tif, dem_dangle_var_ar, area_name + 'dem_dangle_var_confidence.tif')



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
