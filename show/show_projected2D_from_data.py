import os
import sys
import yaml
import json
import argparse
import numpy as np
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
import utils.geometric_utils as gu
import dataset.dataset_utils as du
from dataset.custom_dataset import HPE_Dataset


class projected2DTester():
    def __init__(self, json_file, paramsdir, joint_list, axes_3D):
        self.input_data = json.load(open(json_file, 'rb'))

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

    def get2D(self, graph):
        points2D = {}
        for cam_idx, cam in enumerate(graph.camera_names):
            points2D[cam] = {}
            joint_dictionary = graph._joint_dictionary[cam]
            for joint in joint_dictionary:
                joint_name = str(joint)
                
                #Denormalize graph.x
                data = graph.x[cam_idx, :] * graph.norm_vector
                
                x = data[graph.all_features.index(joint_name + '_x2d')]
                y = data[graph.all_features.index(joint_name + '_y2d')]
                v = data[graph.all_features.index(joint_name + '_visibility')]
                points2D[cam][joint] = (joint_name, x, y, v)
        
        return points2D

    def get2D_orig(self, orig2d):
        tensor_2D = orig2d["coord2D"]
        points2D = {}
        for cam_idx, cam in enumerate(orig2d["available_cams"]):
            points2D[cam] = {}
            for joint in self.joint_list:
                joint_name = str(joint)
                
              
                x = tensor_2D[cam_idx, 0, joint].item()
                y = tensor_2D[cam_idx, 1, joint].item()
                v = tensor_2D[cam_idx, 2, joint].item()
                points2D[cam][joint_name] = (joint, x, y, v)

            print(cam, points2D[cam])
        
        return points2D

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Display 3D multi-pose results using triangulation')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file containing the command argumnets')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    if not config['execution']['training'] and not config['testing']['test_cam_list']:
        raise ValueError("test_cam_lis must have value if dataset is used in testing")

    trainset   = config['paths']['trainset']
    paramsdir  = config['paths']['paramsdir']
    joint_list = config['network']['joint_list']
    axes_3D    = config['visualization']['axes_3D']

    prj_tester = projected2DTester(trainset, paramsdir, joint_list, axes_3D)
    
    dataset = HPE_Dataset(config)
    
    path = str(ROOT) + "/utils/human_pose.json"
    with open(path, 'r') as f:
        human_pose = json.load(f)
        bones = human_pose["skeleton"]

    i = -1

    
    width          = config['visualization']['width']
    height         = config['visualization']['height']
    camera_colours = config['visualization']['camera_colours']

    while True:
        i+=1
        prj_image = np.full((height, width, 3), 255, dtype=np.uint8)
        graph, original2D = dataset[i]

        new_camera_colours={}
        for joint, c in enumerate(graph.camera_names):
            color = camera_colours[list(camera_colours.keys())[joint]]
            new_camera_colours[c] = color
            
        # points2D = prj_tester.get2D_orig(original2D)
        points2D = prj_tester.get2D(graph)
        du.draw2DKeypoints(prj_image, points2D, bones, new_camera_colours)
        cv2.imshow("Projected 2D", prj_image)
        k = cv2.waitKey(0)
        if k%256 == 27:
            print("Escape hit, closing...")
            cv2.destroyAllWindows()
            break

