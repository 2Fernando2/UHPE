import os
import sys
import yaml
import json
import argparse
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
import utils.geometric_utils as gu
from utils.visualizer import Visualizer
from dataset.custom_dataset import HPE_Dataset

class HPETriangulation():
    def __init__(self, config):
        
        self.dataset = HPE_Dataset(config)
        self.frame_map = { f['frame_idx']: i for i, f in enumerate(self.dataset.frames) }
        
        self.get_camera_params()
        
    def get_camera_params(self):
        self.intr_matrices = {}
        self.ext_matrices = {}
        self.distortion_coefficients = {}
        
        for calib_idx in self.dataset.calib_ids:            
            for cam in self.dataset.id_cams[calib_idx]:
                self.intr_matrices[cam] = self.dataset.camera_params[calib_idx][cam]['K']
                self.ext_matrices[cam] = self.dataset.camera_params[calib_idx][cam]['ext']
                self.distortion_coefficients[cam] = self.dataset.camera_params[calib_idx][cam]['distCoef']
        
    def estimate3D(self, iter):
        if iter >= len(self.dataset.frames):
            return None, None
        
        if iter in self.frame_map:
            frame_idx = self.frame_map[iter]
        else:
            return None, None
        
        return [self.dataset.frames[frame_idx]['3D']], ['r']

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Display 3D multi-pose results using triangulation')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file containing the command argumnets')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    if not config['execution']['training'] and not config['testing']['test_cam_list']:
        raise ValueError("test_cam_lis must have value if dataset is used in testing")

    HPE = HPETriangulation(config)

    joint_list  = config['network']['joint_list']
    plot_period = config['visualization']['plot_period']
    data_step   = config['visualization']['data_step']
    axes_3D     = config['visualization']['axes_3D']

    v = Visualizer(plot_period, data_step, HPE, joint_list, axes_3D)
    v.animation()

