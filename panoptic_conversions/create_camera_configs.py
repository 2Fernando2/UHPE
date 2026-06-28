import sys
import os
import json
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create camera parameters\' files from a Panoptic calibration file')
    parser.add_argument('--calibfile', type=str, required=True, help='Panoptic calibration file')
    parser.add_argument('--outputdir', type=str, required=True, help='Output directory')

    args = parser.parse_args()

    panoptic_file = args.calibfile

    os.makedirs(args.outputdir, exist_ok=True)

    panoptic_cam_data = json.load(open(panoptic_file, 'rb'))
    for d in panoptic_cam_data['cameras']:
        cam_name = d['type'] + '_' + d['name']
        d['name'] = cam_name
        cam_file = 'calib_' + cam_name + '.json'
        print(cam_name)
        print('----')
        with open(os.path.join(args.outputdir, cam_file), 'w') as f:
            json.dump(d, f)



