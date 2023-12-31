#/raid-manaslu/amueting/PhD/Project3/PlanetScope_Data/aoi[3-10]/group[1-3]/disparity_maps
# conda activate mintpy
# cd /raid/Planet_NWArg
# mkdir PS2_aoi3
# cd PS2_aoi3
# mkdir disparity_maps
# rsync -avz bodo@manaslu:/raid-manaslu/amueting/PhD/Project3/PlanetScope_Data/aoi3/group[1-3]/disparity_maps/*_polyfit-F.tif disparity_maps
# rsync -avz bodo@manaslu:/raid-manaslu/amueting/PhD/Project3/PlanetScope_Data/aoi3/PSBSD/disparity_maps/*_polyfit-F.tif disparity_maps
# rsync -avz bodo@manaslu:/raid-manaslu/amueting/PhD/Project3/DEM_Data/clips/CopernicusDEM_clip_aoi3.tif .
mkdir /home/bodo/Dropbox/Argentina/aoi3/
export PYTHONPATH=$PYTHONPATH:/home/bodo/Dropbox/soft/github/Planet2mintpy
python /home/bodo/Dropbox/soft/github/Planet2MintPy/create_offset_confidence.py \
           --kernel_size 9 \
           --method 1 \
           --offset_tif_fn "disparity_maps/*_polyfit-F.tif" \
           --area_name aoi3 \
           --npy_out_path npy \
           --confidence_tif_out_path confidence \
           --sensor PS 2>&1 | tee create_offset_confidence.py.log

# python /home/bodo/Dropbox/soft/github/Planet2MintPy/create_offset_confidence.py \
#    --method 2 \
#    --kernel_size 9 \
#    --threshold_angle 45 \
#    --threshold_size 1000 \
#    --offset_tif_fn "disparity_maps/*_polyfit-F.tif" \
#    --area_name aoi3 \
#    --npy_out_path npy2 \
#    --confidence_tif_out_path confidence \
#    --sensor PS 2>&1 | tee create_offset_confidence.py.log


python /home/bodo/Dropbox/soft/github/Planet2MintPy/generate_landslide_mask.py \
          --offset_tif_fn "disparity_maps/*_polyfit-F.tif" \
          --area_name aoi3 \
          --mask_out_path masks \
          --threshold_angle 45 \
          --threshold_size 5000 \
          --out_pngfname aoi3_landslide_mask.png


python /home/bodo/Dropbox/soft/github/Planet2MintPy/prep_Planet_metadata.py \
  --offset_tif_fn "disparity_maps/*_polyfit-F.tif" \
  --dx_confidence_tif_fn "confidence/*_confidence.tif" \
  --dy_confidence_tif_fn "confidence/*_confidence.tif" \
  --mask_tif_fn "confidence/*_mask.tif" \
  --metadata_fn PS2_aoi3_metadata.txt --sensor PS

mkdir mintpy
sed -e 's#METAFILE_FN#/raid/Planet_NWArg/PS2_aoi3/PS2_aoi3_metadata.txt#' \
  -e 's#azOffFile_FN#/raid/Planet_NWArg/PS2_aoi3/disparity_maps/*-F_EW.vrt#' \
  -e 's#rgOffFile_FN#/raid/Planet_NWArg/PS2_aoi3/disparity_maps/*-F_NS.vrt#' \
  -e 's#azOffStdFile_FN#/raid/Planet_NWArg/PS2_aoi3/confidence/*_confidence.tif#' \
  -e 's#rgOffStdFile_FN#/raid/Planet_NWArg/PS2_aoi3/confidence/*_confidence.tif#' \
  -e 's#demFile_FN#/raid/Planet_NWArg/PS2_aoi3/CopernicusDEM_clip_aoi3.tif#' \
  -e 's#mask_FN#/raid/Planet_NWArg/PS2_aoi3/confidence/*_mask.tif#' \
  /home/bodo/Dropbox/soft/github/Planet2MintPy/PS2_Mintpy_template.cfg \
  >/raid/Planet_NWArg/PS2_aoi3/mintpy/PS2_aoi3_config.cfg

python /home/bodo/Dropbox/soft/github/Planet2MintPy/prep_Planet_stack.py \
          --dx_fn "/raid/Planet_NWArg/PS2_aoi3/disparity_maps/*-F_EW.vrt" \
          --dy_fn "/raid/Planet_NWArg/PS2_aoi3/disparity_maps/*-F_NS.vrt" \
          --dx_confidence_fn "/raid/Planet_NWArg/PS2_aoi3/confidence/*_confidence.tif" \
          --dy_confidence_fn "/raid/Planet_NWArg/PS2_aoi3/confidence/*_confidence.tif" \
          --mask_fn "confidence/*_mask.tif" \
          --meta_file /raid/Planet_NWArg/PS2_aoi3/PS2_aoi3_metadata.txt --pixel_size 3.0 \
          --template_file /raid/Planet_NWArg/PS2_aoi3/mintpy/PS2_aoi3_config.cfg \
          --h5_stack_fn /raid/Planet_NWArg/PS2_aoi3/mintpy/inputs/geo_offsetStack_aoi3.h5

cd /raid/Planet_NWArg/PS2_aoi3/mintpy/
ifgram_inversion.py /raid/Planet_NWArg/PS2_aoi3/mintpy/inputs/geo_offsetStack_aoi3.h5 \
    --skip-reference -i azimuthOffset --md offsetSNR --mt 0.5 -w var -c local --num-worker 20 --mem 16 -o timeseriesAz_var.h5 residualInvAz_var.h5 numInvOffsetAz_var.h5
ifgram_inversion.py /raid/Planet_NWArg/PS2_aoi3/mintpy/inputs/geo_offsetStack_aoi3.h5 \
    --skip-reference -i rangeOffset --md offsetSNR --mt 0.5 -w var -c local --num-worker 20 --mem 16 -o timeseriesRg_var.h5 residualInvRg_var.h5 numInvOffsetRg_var.h5

view.py timeseriesAz_var.h5 --nodisplay --vlim -2 2 -u m --title "PS2 var: Az (NS) timeseries" -c RdYlBu --save -o PS2_ts_NS_var.png
view.py timeseriesRg_var.h5 --nodisplay --vlim -2 2 -u m --title "PS2 var: Rg (EW) timeseries" -c RdYlBu --save -o PS2_ts_EW_var.png
convert -quality 50 PS2_ts_NS_var.png /home/bodo/Dropbox/Argentina/aoi3/PS2_aoi3_ts_NS_var.jpg
convert -quality 50 PS2_ts_EW_var.png /home/bodo/Dropbox/Argentina/aoi3/PS2_aoi3_ts_EW_var.jpg

timeseries2velocity.py --ref-yx 100 50 timeseriesAz_var.h5 -o velocityAz_var.h5
timeseries2velocity.py --ref-yx 100 50 timeseriesRg_var.h5 -o velocityRg_var.h5

view.py velocityAz_var.h5 velocity --nodisplay --title "PS2 var: Mean Velocity Az (NS)" --zero-mask -u meters --vlim -1 1 -c RdYlBu --save -o PS2_NS_var_nomask_meanv.png
view.py velocityAz_var.h5 --nodisplay --title "PS2 var: Stats Az (NS)" --zero-mask -u meters --vlim -1 1 -c RdYlBu --save -o PS2_NS_var_nomask_stats.png
convert -quality 50 PS2_NS_var_nomask_meanv.png /home/bodo/Dropbox/Argentina/aoi3/PS2_aoi3_NS_var_nomask_meanv.jpg
convert -quality 50 PS2_NS_var_nomask_stats.png /home/bodo/Dropbox/Argentina/aoi3/PS2_aoi3_NS_var_nomask_stats.jpg

view.py velocityRg_var.h5 velocity --nodisplay --title "PS2 var: Mean Velocity Rg (EW)" --zero-mask -u meters --vlim -1 1 -c RdYlBu --save -o PS2_EW_var_nomask_meanv.png
view.py velocityRg_var.h5 --nodisplay --title "PS2 var: Stats Rg (EW)" --zero-mask -u meters --vlim -1 1 -c RdYlBu --save -o PS2_EW_var_nomask_stats.png
convert -quality 50 PS2_EW_var_nomask_meanv.png /home/bodo/Dropbox/Argentina/aoi3/PS2_aoi3_EW_var_nomask_meanv.jpg
convert -quality 50 PS2_EW_var_nomask_stats.png /home/bodo/Dropbox/Argentina/aoi3/PS2_aoi3_EW_var_nomask_stats.jpg

python /home/bodo/Dropbox/soft/github/Planet2MintPy/calc_velocity_from_AzRg.py \
    --rg_file /raid/Planet_NWArg/PS2_aoi3/mintpy/velocityRg_var.h5 \
    --az_file /raid/Planet_NWArg/PS2_aoi3/mintpy/velocityAz_var.h5 \
    --HDF_outfile /raid/Planet_NWArg/PS2_aoi3/mintpy/aoi3_var_velocity.h5 \
    --mask00 False

view.py aoi3_var_velocity.h5 vel_magnitude -u meters --vlim 0 4 --nodisplay \
  --title "PS2 var: Velocity Magnitude" --zero-mask --save -o PS2_var_vel_magnitude.png
view.py aoi3_var_velocity.h5 vel_direction --vlim 0 360 -c hsv --nodisplay \
  --title "PS2 var: Velocity Direction" --save -o PS2_var_vel_direction.png

# generate_mask.py aoi3_var_velocity.h5 vel_magnitude -m 1 -p 10000 -o aoi3_var_velocity_mask.h5
# view.py aoi3_var_velocity.h5 vel_magnitude --mask aoi3_var_velocity_mask.h5 -u meters \
#   --vlim 1 4 --nodisplay --title "PS2 var: Velocity Magnitude" --zero-mask --save -o PS2_var_vel_magnitude_mask.png

# view.py aoi3_var_velocity.h5 vel_magnitude --mask aoi3_var_velocity_mask.h5 -u meters --vlim 3 6 --nodisplay --title "PS2 var: Velocity Magnitude" --zero-mask --save -o PS2_var_vel_magnitude_mask.png
# view.py aoi3_var_velocity.h5 vel_direction --mask aoi3_var_velocity_mask.h5 --vlim 0 360 -c hsv --nodisplay --title "PS2 var: Velocity Direction" --save -o PS2_var_vel_direction_mask.png
# extract_mask_TS.py \
#     --az_ts_file /raid/Planet_NWArg/PS2_aoi3/mintpy/timeseriesAz_var.h5 \
#     --rg_ts_file /raid/Planet_NWArg/PS2_aoi3/mintpy/timeseriesRg_var.h5 \
#     --mask_file /raid/Planet_NWArg/PS2_aoi3/mintpy/aoi3_var_velocity_mask.h5 \
#     --HDF_outfile  /raid/PS2_aoi6/mintpy/aoi3_var_velocity_mask_ts.h5 \
#     --npy_outfile /raid/PS2_aoi6/mintpy/aoi3_var_velocity_mask_ts.npy \
#     --out_pngfname /raid/PS2_aoi6/mintpy/aoi3_var_velocity_mask_ts.png
