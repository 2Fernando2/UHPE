import os
import sys
import yaml
import json
import torch
import argparse
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from utils.models import GATNetwork, RGCNetwork, GCNetwork
from utils.visualizer import Visualizer
from dataset.custom_dataset import HPE_Dataset, hpe_collate_fn

class HPEModel():
    def __init__(self, config):
        
        self.dataset = HPE_Dataset(config)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        json_files = config['paths']['trainset']

        # Extraxt all the json_files paths
        p = Path(json_files)
        root_dir = p.parent
        pattern = p.name
        json_files = list(root_dir.glob(pattern))
        self.input_data = []

        #Ensure json_files is a list
        if isinstance(json_files, str):
            json_files = [json_files]        

        for json_file in json_files:
            print(json_file)
            self.input_data = json.load(open(json_file, 'rb')) 
        
        
        star_topology   = config['network']['star_topology']
        self.joint_list = config['network']['joint_list']
        checkpoint = torch.load(config['network']['model_pth'], map_location=torch.device('cpu'))
        in_channels  = checkpoint['config']['in_channels']
        out_channels = checkpoint['config']['out_channels']
        if not 'model_type' in checkpoint['config']:
            model_type = "GATNetwork"
        else:
            model_type   = checkpoint['config']['model_type']


        if model_type == "GATNetwork":
            self.model = GATNetwork(in_channels, out_channels, checkpoint['config'], star_topology).to(self.device)
        elif model_type == "RGCNetwork":
            self.model = RGCNetwork(in_channels, out_channels, checkpoint['config'], star_topology).to(self.device)
        elif model_type == "GCNetwork":
            self.model = GCNetwork(in_channels, out_channels, checkpoint['config'], star_topology).to(self.device)
        else:
            raise ValueError(f"Error: Model type '{model_type}' not recognized for saving.")
            
            
        self.frame_map = { i:f['frame_idx'] for i, f in enumerate(self.dataset.frames) }
        
        self.model.load_state_dict(checkpoint['state'])
        self.model.eval()
        
    def estimate3D(self, iter):
        if iter >= len(self.dataset.frames):
            return None, None
        
        # Load data (batch_size = 1)
        
        if iter in self.frame_map:
            frame_idx = self.frame_map[iter]
        else:
            return None, None
            
        graph, _ = self.dataset[iter]
        batch_graph, _ = hpe_collate_fn([(graph, _)])
        batch_graph = batch_graph.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(batch_graph) # [batch_size, 3 * num_joints]
        
        # Reshape to [num_joints, 3]
        coords = outputs[0].cpu().numpy().reshape(-1, 3) # [num_joints, 3]
        
        results = []
        colors = []
        
        # Predicted skeleton
        result3d = {}
        for idx, joint in enumerate(self.dataset.joint_list):
            joint_name = str(joint)
            result3d[joint_name] = coords[idx].reshape(3, 1) # np.array([[x], [y], [z]])

        results.append(result3d)
        colors.append('r')
        
        # Ground truth skeleton
        input_element = self.input_data[frame_idx]
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Display 3D multi-pose results using triangulation')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file containing the command argumnets')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    if not config['execution']['training'] and not config['testing']['test_cam_list']:
        raise ValueError("test_cam_lis must have value if dataset is used in testing")
    
    HPE = HPEModel(config)

    joint_list  = config['network']['joint_list']
    plot_period = config['visualization']['plot_period']
    data_step   = config['visualization']['data_step']
    axes_3D     = config['visualization']['axes_3D']

    v = Visualizer(plot_period, data_step, HPE, joint_list, axes_3D)
    v.animation()

