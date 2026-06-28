import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
import utils.geometric_utils as gu
from utils.visualizer import Visualizer


class HPETriangulation():
    def __init__(self, json_file, paramsdir, joint_list, axes_3D):
        self.input_data = json.load(open(json_file, 'rb'))

        self.paramsdir = paramsdir
        self.joint_list = joint_list
        self.axes_3D = axes_3D

        self.get_camera_params()

    def get_camera_params(self):
        #Get camera names
        self.camera_names = []
        for cams_data in self.input_data:
            for cam in cams_data:
                if cam not in self.camera_names:
                    self.camera_names.append(cam)
    
        self.camera_params = {}
        self.camera_matrices = {}
        self.distortion_coefficients = {}
        self.projection_matrices = {}
        self.ext_matrices = {}
        for cam in self.camera_names:
            params_file = os.path.join(self.paramsdir, 'calib_'+cam+'.json')
            cam_params = json.load(open(params_file, 'rb'))
            self.camera_params[cam] = gu.get_parameters(cam_params)
            self.camera_matrices[cam] = self.camera_params[cam]['K']
            self.distortion_coefficients[cam] = self.camera_params[cam]['distCoef']
            self.projection_matrices[cam] = self.camera_params[cam]['proj']
            self.ext_matrices[cam] = self.camera_params[cam]['ext']
            
    def estimate3D(self, itert):
        if itert>=len(self.input_data):
            return None, None
        
        results = []
        colors = []

        input_element = self.input_data[itert]
        joints_data = dict()
        for cam in input_element:
            cam_data = json.loads(input_element[cam][0])[0] # assuming only one human
            for j in cam_data:
                if cam_data[j][3] > 0.5:
                    if not j in joints_data.keys():
                        joints_data[j] = {}
                    joints_data[j][cam] = [cam_data[j][1], cam_data[j][2]]

        result3D = gu.triangulate(joints_data, self.joint_list, self.camera_matrices, self.distortion_coefficients, self.ext_matrices, self.axes_3D['Y'][0]) 

        results.append(result3D)
        colors.append('r')

        # result3D_cam = self.transform3D_to_camera_view(self.ext_matrices[self.camera_names[0]], result3D)

        # Ground truth skeleton
        input_element = self.input_data[itert]
        first_cam = list(input_element.keys())[0]
        if len(input_element[first_cam]) != 4:
            print("There is no ground truth in the specified file")
            exit()
        
        joints_3D_all = input_element[first_cam][3]
        result3d_gt = {}

        for joints_3D in joints_3D_all:
            for joint in self.joint_list:
                joint_name = str(joint)
                if joint_name in joints_3D:
                    result3d_gt[joint_name] = np.array(joints_3D[joint_name]).reshape(3, 1)
            
            results.append(result3d_gt)
            colors.append('d')

        return results, colors
    
    def transform3D_to_camera_view(self, Me, points3D_dict):
        points3D = list(points3D_dict.values())
        if not points3D:
            return {}
        L = len(points3D)
        if L>1:
            points3D = np.concatenate(points3D, axis=1)
        else:
            points3D = points3D[0]
        
        points3D_cam = gu.points_from_FR1_to_FR2(Me, points3D)
        points3D_cam_dict = {}
        for j, kp in enumerate(list(points3D_dict.keys())):
            points3D_cam_dict[kp] = points3D_cam[:,j].reshape((3,1))

        return points3D_cam_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Display 3D multi-pose results using triangulation')

    parser.add_argument('--testfile', type=str, required=True, help='Test file used as input')
    parser.add_argument('--paramsdir', type=str, nargs='?', required=True, help='Directory that contains the camera parameters files')
    parser.add_argument('--plotperiod', type=int, nargs='?', required=False, default=10, help='Plot period (miliseconds)')
    parser.add_argument('--datastep', type=int, nargs='?', required=False, default=10, help='Data step used to plot the results')

    args = parser.parse_args()

    TEST_FILE = args.testfile

    PLOTPERIOD = args.plotperiod  # In miliseconds
    DATASTEP = args.datastep
    PARAMSDIR = args.paramsdir

    joint_list = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    axes_3D = {'X': (0, 1.), 'Y': (2, 1.), 'Z': (1, -1.)}

    HPE = HPETriangulation(TEST_FILE, PARAMSDIR, joint_list, axes_3D)

    v = Visualizer(PLOTPERIOD, DATASTEP, HPE, joint_list, axes_3D)
    v.animation()

