#!/usr/bin/env python

############################################################
# Created by Bodo Bookhagen                                   #
############################################################

import glob, os, csv, sys, subprocess, argparse

EXAMPLE = """example:
  prep_Planet_metadata.py --offset_tif_fn "disparity_maps/*_polyfit-F.tif" \
          --dx_confidence_tif_fn "confidence/*_confidence.tif" \
          --dy_confidence_tif_fn "confidence/*_confidence.tif" \
          --mask_tif_fn "confidence/*_mask.tif" \
          --metadata_fn PS2_aoi3_metadata.txt --sensor PS2
"""

DESCRIPTION = """
Prepare metadata from PlanetScope for inversion with mintpy. Reads in the tif files generated during the correlation, extracts dx and dy offsets and generates virtual files for further processing (e.g., loading into a mintpy container).

Aug-2023, Bodo Bookhagen (bodo.bookhagen@uni-potsdam.de) and Ariane Mueting (mueting@uni-potsdam.de)
"""

def cmdLineParser():
    from argparse import RawTextHelpFormatter
    parser = argparse.ArgumentParser(description=DESCRIPTION, epilog=EXAMPLE, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--offset_tif_fn', help='List of tif files containing dx and dy offsets corresponding to bands 1 and 2', required=True)
    parser.add_argument('-m', '--metadata_fn', help='Output text file containing metadata (to be read by mintpy)', required=True)
    parser.add_argument('--sensor', default='PS', help='Chose sensor among L8 and PS. Will determine the naming structure of file.', required=False)
    parser.add_argument('--dx_confidence_tif_fn', default='', help='List of tif files containing confidence values for dx (same data can be used for dx and dy if required)', required=False)
    parser.add_argument('--dy_confidence_tif_fn', default='', help='List of tif files containing confidence values for dy. (same data can be used for dx and dy if required)', required=False)
    parser.add_argument('--mask_tif_fn', default='', help='List of tif files containing mask for each time step (same for dx and dy)', required=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = cmdLineParser()

    filelist = glob.glob(args.offset_tif_fn)
    filelist.sort()
    gdal_translate_command = []
    for i in range(len(filelist)):
        cfile = os.path.basename(filelist[i])
        # create virtual files containing first band (X: EW) and second band (Y: NS)
        gdal_translate_command.append('gdal_translate -q -b 1 %s %s'%(filelist[i], filelist[i][:-4]+'_EW.vrt'))
        gdal_translate_command.append('gdal_translate -q -b 2 %s %s'%(filelist[i], filelist[i][:-4]+'_NS.vrt'))


    gdal_translate_command_fn = args.metadata_fn + '_gdal_translate_command.sh'
    with open(gdal_translate_command_fn, 'w') as f:
        for line in gdal_translate_command:
            f.write("%s\n" % line)

    print("Calling gdal_translate and convert all offset tif files to vrt")
    subprocess.call(['sh', gdal_translate_command_fn])

    #now generate input for metadata file: List of date1, date2, filename (also for confidence)
    filelist2a = glob.glob(os.path.join(os.path.dirname(filelist[0]), '*_EW.vrt'))
    filelist2b = glob.glob(os.path.join(os.path.dirname(filelist[0]), '*_NS.vrt'))
    #filelist2c = glob.glob(os.path.join(os.path.dirname(filelist[0]), os.path.basename(tif_filelist)[:-4] + '_mask.vrt'))

    filelist2 = filelist2a + filelist2b
    metadata_list = []
    for i in range(len(filelist2)):
        cfile = os.path.basename(filelist2[i])
        date0 = cfile.split('_')[0]
        time0 = cfile.split('_')[1]
        if args.sensor == 'L8':
            date1 = cfile.split('_')[1]
            time0 = '0'
            time1 = '0'
        if args.sensor == 'PS':
            #need to distinguish between PSBSD and PS2 scene IDs
            if len(cfile.split('_')[3]) == 8:
                date1 = cfile.split('_')[3]
                time1 = cfile.split('_')[4]
            else: 
                date1 = cfile.split('_')[4]
                time1 = cfile.split('_')[5]
                

        metadata_list.append([cfile, date0, date1])

    filelist3 = []
    if len(args.dx_confidence_tif_fn) > 0:
        filelist_dx_confidence = glob.glob(args.dx_confidence_tif_fn)
        filelist_dx_confidence.sort()
        if len(args.dy_confidence_tif_fn) == 0:
            filelist3 = filelist_dx_confidence
            print('Found only dx confidence values')
    if len(args.dy_confidence_tif_fn) > 0:
        filelist_dy_confidence = glob.glob(args.dy_confidence_tif_fn)
        filelist_dy_confidence.sort()
        if len(args.dx_confidence_tif_fn) > 0:
            filelist3 = filelist_dx_confidence + filelist_dy_confidence
            print('Found dx and dy confidence values and creating list')
        else:
            filelist3 = filelist_dy_confidence
            print('Found only dy confidence values')

    if len(filelist3) > 0:
        for i in range(len(filelist3)):
            cfile = os.path.basename(filelist3[i])
            date0 = cfile.split('_')[1]
            if args.sensor == 'L8':
                date1 = cfile.split('_')[2]
            elif args.sensor == 'PS':
                #need to distinguish between PSBSD and PS2 scene IDs
                if len(cfile.split('_')[3]) == 8:
                    date1 = cfile.split('_')[3]
                else: 
                    date1 = cfile.split('_')[4]

            metadata_list.append([cfile, date0, date1])


    filelist4 = []
    if len(args.mask_tif_fn) > 0:
        filelist_mask = glob.glob(args.mask_tif_fn)
        filelist_mask.sort()
        if len(args.mask_tif_fn) == 0:
            filelist4 = filelist_mask
            print('Found mask tif file')
    if len(filelist4) > 0:
        for i in range(len(filelist4)):
            cfile = os.path.basename(filelist4[i])
            date0 = cfile.split('_')[1]
            if args.sensor == 'L8':
                date1 = cfile.split('_')[2]
            elif args.sensor == 'PS':
                #need to distinguish between PSBSD and PS2 scene IDs
                if len(cfile.split('_')[3]) == 8:
                    date1 = cfile.split('_')[3]
                else: 
                    date1 = cfile.split('_')[4]

            metadata_list.append([cfile, date0, date1])

    # write metadata to file
    with open(args.metadata_fn, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=' ')
        writer.writerows(metadata_list)
