import os
import sys
import yaml
import json
import argparse
import numpy as np
from pathlib import Path
import cv2
import random

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
import utils.geometric_utils as gu
import dataset.dataset_utils as du
from dataset.custom_dataset import HPE_Dataset

class projected2DTester_dataset():
    def __init__(self, json_file, paramsdir, joint_list, axes_3D):
        self.input_data = json.load(open(json_file[0], 'rb'))
        
        self.paramsdir = paramsdir
        self.joint_list = joint_list
        self.axes_3D = axes_3D
        
        self.get_camera_params()
        
    def get_camera_params(self):
        #Get camera names
        self.camera_names = []
        camera_files = os.listdir(self.paramsdir)
        for f in camera_files:
            if f.startswith('calib_') and f.endswith('.json'):
                cam = f.split('.json')[0].split('calib_')[-1]
                self.camera_names.append(cam)
                
        self.camera_params = {}
        self.camera_matrices = {}
        self.distorsion_coefficients = {}
        self.projection_matrices = {}
        self.ext_matrices = {}
        self.resolution = {}
        for cam in self.camera_names:
            params_file = os.path.join(self.paramsdir, 'calib_'+cam+'.json')
            cam_params = json.load(open(params_file, 'rb'))
            self.camera_params[cam] = gu.get_parameters(cam_params)
            self.camera_matrices[cam] = self.camera_params[cam]['K']
            self.distorsion_coefficients[cam] = self.camera_params[cam]['distCoef']
            self.projection_matrices[cam] = self.camera_params[cam]['proj']
            self.ext_matrices[cam] = self.camera_params[cam]['ext']
            self.resolution[cam] = self.camera_params[cam]['resolution']
            
    def estimate3D(self, itert):
        if itert>=len(self.input_data):
            return None
        
        frame = self.input_data[itert]
        joints_data = dict()
        for cam in frame:
            cam_data = json.loads(frame[cam][0])[0] # assuming only one human
            for j in cam_data:
                if cam_data[j][3] > 0.5:
                    if not j in joints_data.keys():
                        joints_data[j] = {}
                    joints_data[j][cam] = [cam_data[j][1], cam_data[j][2]]

        result3D = gu.triangulate(joints_data, self.joint_list, self.camera_matrices, self.distorsion_coefficients, self.ext_matrices, self.axes_3D['Y'][0]) 
        return result3D

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Display 3D multi-pose results using triangulation')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file containing the command argumnets')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    if not config['execution']['training'] and not config['testing']['test_cam_list']:
        raise ValueError("test_cam_lis must have value if dataset is used in testing")

    
    trainset       = config['paths']['trainset']
    paramsdir      = config['paths']['paramsdir']
    joint_list     = config['network']['joint_list']
    axes_3D        = config['visualization']['axes_3D']
    camera_colours = config['visualization']['camera_colours']
    
    prj_tester = projected2DTester_dataset(trainset, paramsdir, joint_list, axes_3D)
    
    dataset = HPE_Dataset(config)    
    
    new_cams = random.sample(prj_tester.camera_names, k=5)
    new_camera_colours={}
    for i, c in enumerate(new_cams):
        color = camera_colours[list(camera_colours.keys())[i]]
        new_camera_colours[c] = color

    path = str(ROOT) + "/utils/human_pose.json"
    with open(path, 'r') as f:
        human_pose = json.load(f)
        bones = human_pose["skeleton"]

    i = -1

    
    width  = config['visualization']['width']
    height = config['visualization']['height']

    while True:
        i+=1
        calib_idx = dataset.frames[i]['calib_idx']
        prj_image = np.full((height, width, 3), 255, dtype=np.uint8)
        estimated3D = prj_tester.estimate3D(i)
        if  estimated3D is not None:
            projected2D = du.generate_new_data(estimated3D, new_cams, dataset.camera_params[calib_idx])
            if projected2D is not None:
                du.draw2DKeypoints(prj_image, projected2D, bones, new_camera_colours)
                cv2.imshow("Projected 2D", prj_image)
                k = cv2.waitKey(0)
                if k%256 == 27:
                    print("Escape hit, closing...")
                    cv2.destroyAllWindows()
                    break

