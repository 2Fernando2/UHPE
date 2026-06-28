import os
import sys
import yaml
import json
import argparse
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from utils.visualizer import Visualizer
from dataset.custom_dataset import HPE_Dataset

class HPETriangulation():
    def __init__(self, config):
        self.joint_list = config['network']['joint_list']
        self.dataset = HPE_Dataset(config)

    def estimate3D(self, iter):
        if iter >= len(self.dataset.frames):
            return None
        
        graph, original2D = self.dataset[iter]
        result3d = {}
        for cam_idx, cam in enumerate(graph.camera_names):
            for joint in self.joint_list:
                joint = str(joint)
                
                #Denormalize graph.x
                data = graph.x[cam_idx, :] * graph.norm_vector
                
                estimated = data[graph.all_features.index(joint + '_estimated')]
                if estimated != 0.0:
                    result3d[joint] = np.array([[data[graph.all_features.index(joint + '_x3d')].item()],
                                                [data[graph.all_features.index(joint + '_y3d')].item()],
                                                [data[graph.all_features.index(joint + '_z3d')].item()]])
        # print(result3d)                
        return [result3d], ['r']

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

