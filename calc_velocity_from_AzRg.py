#!/usr/bin/env python
import warnings, argparse, os, tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mintpy.utils import readfile, writefile
from osgeo import gdal

from scipy import signal
EXAMPLE = """example:
python /home/bodo/Dropbox/Argentina/Planet2mintpy/calc_velocity_from_AzRg.py \
    --rg_file /raid/PS2_aoi3/mintpy/velocityRg_var.h5 \
    --az_file /raid/PS2_aoi3/mintpy/velocityAz_var.h5 \
    --HDF_outfile /raid/PS2_aoi3/mintpy/aoi3_var_velocity.h5 \
    --mask00 False
"""

DESCRIPTION = """
Take range (Rg) and azimut (Az) velocities from timeseries2velocity.py and calculated velocity magnitude and direction. Writes output to new HDF file and also will create mask (if threshold is given).

Aug-2023, Bodo Bookhagen (bodo.bookhagen@uni-potsdam.de) and Ariane Mueting (mueting@uni-potsdam.de)
"""

def cmdLineParser():
    from argparse import RawTextHelpFormatter
    parser = argparse.ArgumentParser(description=DESCRIPTION, epilog=EXAMPLE, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--rg_file', help='Rg offset', required=True)
    parser.add_argument('--az_file', help='Az offset', required=True)
    parser.add_argument('--HDF_outfile', help='Output velocity filename with magnitude and direction as HDF', required=True)
    parser.add_argument('--vel_mask', type=float, default=0.0, help='Mask velocity magnitude with value', required=False)
    parser.add_argument('--dem_fn', help='DEM in same size as offset files', required=False)
    parser.add_argument('--dem_delta_aspect', default=0.0, type=float, help='Mask velocity direction with DEM aspect. This gives degree difference from aspect that is allowed.', required=False)
    parser.add_argument('--mask00', type=bool, default=False, help='Use value from [0,0] and set to nan', required=False)
#    parser.add_argument('--HDF_mask_outfile', help='Output mask filename', required=False)
#    parser.add_argument('--out_pngfname', default="Escoipe_LDS_epsg4326_vUP_Violinplot.png", type=str, help='Output filename for PNG file', required=True)
    return parser.parse_args()


def aspect_slope_dem(dem_fname, kernel_size = 3):
    # Takes DEM (will need to be same size as correlation images - same nrows and ncols)
    # calculates aspect and slope within window given by kernel_size.
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
    slope = np.float32(np.rad2deg(np.arctan(np.sqrt(dem_y**2+dem_x**2) / gt[1])))
    aspect = np.float32(np.rad2deg(np.arctan2(dem_y, dem_x)))

    aspect.ravel()[aspect.ravel() < 0] = np.abs(aspect.ravel()[aspect.ravel() < 0]) + 180.
    aspect = aspect - 90.
    aspect.ravel()[aspect.ravel() < 0] = np.abs(aspect.ravel()[aspect.ravel() < 0]) + 90.

    return dem, slope, aspect


if __name__ == '__main__':
    args = cmdLineParser()

#    parser = argparse.ArgumentParser(description='Take Rg and Az mean offsets and calculate velocity magnitude and direction.')
#    args = parser.parse_args()
#    args.rg_file = '/raid/L8_DelMedio/mintpy/velocityRg_var.h5'
#    args.az_file = '/raid/L8_DelMedio/mintpy/velocityAz_var.h5'
#    args.mask = '/home/bodo/Dropbox/Argentina/Sentinel1A/LDS-decomposition/geo_sds1_epsg4326_v_mask.npz'
#    args.HDF_outfile = '/raid/L8_DelMedio/mintpy/DelMedio_var_velocity.h5'
#    args.HDF_mask_outfile = '/raid/L8_DelMedio/mintpy/DelMedio_var_velocity_mask.h5'
#    args.dem_fn='/home/bodo/Dropbox/Argentina/DelMedio/output_COP30_matched_size.tif'
#    args.mask00 = False
#    args.dem_delta_aspect = 45
#    args.vel_mask = 3

    #Rg is dx, Az is dy
    dx = readfile.read(args.rg_file, datasetName='velocity')[0]
    dy = readfile.read(args.az_file, datasetName='velocity')[0]
    data_dict = readfile.read(args.az_file, datasetName='velocity')[1]
    atr = readfile.read_attribute(args.az_file)

    # could add masking step
    # dx[abs(dx) < 0.01] = np.nan
    # dy[abs(dy) < 0.01] = np.nan

    # use value from 0,0 to set to nan
    if args.mask00 == True:
        dx[dx == dx[0,0]] = np.nan
        dy[dy == dy[0,0]] = np.nan

    #dx and dy are still in units of pixels, convert to m
    spatial_resolution = abs(float(atr['X_STEP']))
    dx = dx * spatial_resolution
    dy = dy * spatial_resolution
    vel_magnitude = np.sqrt(dx**2 + dy**2)
    vel_direction = np.rad2deg(np.arctan2(dy, dx))
    # adjust direction: 0 is pointing East, but we want a cardinal system with North Up = 0
    vel_direction.ravel()[vel_direction.ravel() < 0] = np.abs(vel_direction.ravel()[vel_direction.ravel() < 0]) + 180.
    vel_direction = vel_direction - 90.
    vel_direction.ravel()[vel_direction.ravel() < 0] = np.abs(vel_direction.ravel()[vel_direction.ravel() < 0]) + 90.

    ds_data_dict = dict(atr)
    ds_data_dict = {'vel_magnitude': [np.float32, vel_magnitude.shape],
                    'vel_direction': [np.float32, vel_direction.shape]}
    ds_unit_dict = {}
    ds_unit_dict['vel_magnitude'] = 'm/y'
    ds_unit_dict['vel_direction'] = 'degree'

    writefile.layout_hdf5(args.HDF_outfile, metadata=data_dict,
                          ds_name_dict=ds_data_dict, ds_unit_dict=ds_unit_dict)
    writefile.write_hdf5_block(args.HDF_outfile, data=vel_magnitude, datasetName='vel_magnitude', print_msg=False)
    writefile.write_hdf5_block(args.HDF_outfile, data=vel_direction, datasetName='vel_direction', print_msg=False)

    if args.vel_mask > 0:
        #mask velocity with magnitude
        vel_magnitude_masked = vel_magnitude.copy()
        vel_magnitude_masked[vel_magnitude_masked < args.vel_mask] = np.nan
        vel_mask1 = np.zeros(vel_magnitude_masked.shape, dtype=bool)
        vel_mask1[vel_magnitude >= args.vel_mask] = 1
        vel_direction_masked = vel_direction.copy()
        vel_direction_masked[vel_magnitude < args.vel_mask] = np.nan

        if args.dem_delta_aspect > 0 and len(args.dem_fn) > 1:
            #mask with aspect from dem
            dem, dem_slope, dem_aspect = aspect_slope_dem(args.dem_fn, kernel_size = 3)
            direction_difference = np.abs(vel_direction - dem_aspect)
            vel_mask2 = np.zeros(vel_magnitude_masked.shape, dtype=bool)
            vel_mask2[direction_difference <= args.dem_delta_aspect] = 1
            vel_mask = vel_mask1 & vel_mask2
            vel_direction_masked[direction_difference > args.dem_delta_aspect] = np.nan
            vel_magnitude_masked[direction_difference > args.dem_delta_aspect] = np.nan

            # plot aspect, velocity direction, and mask
            fig, ax = plt.subplots(1, 3, figsize=(16, 6), dpi=300)
            im0 = ax[0].imshow(dem_aspect, cmap='hsv', vmin=0, vmax=360)
            ax[0].set_title('DEM aspect', fontsize=14)
            cb0 = fig.colorbar(im0, ax=ax[0], location='bottom', pad=0.1)
            cb0.set_label('DEM Aspect')
            im1 = ax[1].imshow(vel_direction, cmap='hsv', vmin=0, vmax=360)
            ax[1].set_title('Velocity Direction', fontsize=14)
            cb1 = fig.colorbar(im1, ax=ax[1], location='bottom', pad=0.1)
            cb1.set_label('Velocity direction (degree)')
            im2 = ax[2].imshow(vel_direction_masked, cmap='hsv', vmin=0, vmax=360)
            ax[2].set_title('Velocity direction - masked', fontsize=14)
            cb2 = fig.colorbar(im2, ax=ax[2], location='bottom', pad=0.1)
            cb2.set_label('Velocity direction (degree)')
            fig.tight_layout()
            fig.savefig(args.dem_fn + '.direction_mask.png', dpi=300)

            # plot dem output
            fig, ax = plt.subplots(1, 3, figsize=(16, 6), dpi=300)
            im0 = ax[0].imshow(dem, cmap='viridis', vmin=np.nanpercentile(dem,2), vmax=np.nanpercentile(dem, 98))
            ax[0].set_title('DEM Height', fontsize=14)
            cb0 = fig.colorbar(im0, ax=ax[0], location='bottom', pad=0.1)
            cb0.set_label('Elevation')
            im1 = ax[1].imshow(dem_aspect, cmap='hsv', vmin=0, vmax=360)
            ax[1].set_title('DEM Aspect', fontsize=14)
            cb1 = fig.colorbar(im1, ax=ax[1], location='bottom', pad=0.1)
            cb1.set_label('DEM Aspect (degree)')
            im2 = ax[2].imshow(dem_slope, cmap='magma', vmin=0)
            ax[2].set_title('DEM Slope', fontsize=14)
            cb2 = fig.colorbar(im2, ax=ax[2], location='bottom', pad=0.1)
            cb2.set_label('DEM Slope (degree)')
            fig.tight_layout()
            fig.savefig(args.dem_fn + '.png', dpi=300)
        else:
            vel_mask = vel_mask1

        ds_data_dict = dict(atr)
        ds_data_dict = {'vel_magnitude': [np.float32, vel_magnitude.shape],
                        'vel_direction': [np.float32, vel_direction.shape]}
        ds_unit_dict = {}
        ds_unit_dict['vel_magnitude'] = 'm/y'
        ds_unit_dict['vel_direction'] = 'degree'

        writefile.layout_hdf5(args.HDF_outfile[:-3] + '_masked.h5', metadata=data_dict,
                              ds_name_dict=ds_data_dict, ds_unit_dict=ds_unit_dict)
        writefile.write_hdf5_block(args.HDF_outfile[:-3] + '_masked.h5', data=vel_magnitude_masked, datasetName='vel_magnitude', print_msg=False)
        writefile.write_hdf5_block(args.HDF_outfile[:-3] + '_masked.h5', data=vel_direction_masked, datasetName='vel_direction', print_msg=False)

        ds_data_dict = dict(atr)
        ds_data_dict = {'mask': [bool, vel_magnitude.shape]}
        ds_unit_dict = {}
        ds_unit_dict['mask'] = ''
        writefile.layout_hdf5(args.HDF_outfile[:-3] + '_mask.h5', metadata=data_dict,
                              ds_name_dict=ds_data_dict, ds_unit_dict=ds_unit_dict)
        writefile.write_hdf5_block(args.HDF_outfile[:-3] + '_mask.h5', data=vel_mask, datasetName='mask', print_msg=False)
