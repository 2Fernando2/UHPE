import sys
import copy
import json
import time
import yaml
import torch
import pickle
import argparse
import itertools
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
dataset_path = ROOT / "utils"
sys.path.append(str(ROOT))

from utils.models import GATNetwork, RGCNetwork, GCNetwork
import dataset.dataset_utils as du
from dataset.custom_dataset import HPE_Dataset, hpe_supervised_collate_fn
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Print accuracy and time metrics of the GATNetwork model')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file containing the command arguments')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    torch.set_grad_enabled(False)
    
    TRAINSET = config['paths']['trainset']
    
    # Load model
    checkpoint = torch.load(config['network']['model_pth'], map_location=device)    
    in_channels   = checkpoint['config']['in_channels']
    out_channels  = checkpoint['config']['out_channels']
    model_type    = checkpoint['config']['model_type']
    star_topology = config['network']['star_topology']

    if model_type == "GATNetwork":
        model = GATNetwork(in_channels, out_channels, checkpoint['config'], star_topology).to(device)
    elif model_type == "RGCNetwork":
        model = RGCNetwork(in_channels, out_channels, checkpoint['config'], star_topology).to(device)
    elif model_type == "GCNetwork":
        model = GCNerwork(in_channels, out_channels, checkpoint['config'], star_topology).to(device)
    else:
        raise ValueError(f"Error: Model type '{model_type}' not recognized for saving.")

    
    model.load_state_dict(checkpoint['state']) 
    model.eval()
    
    joint_list = config['network']['joint_list']
    use_3D_features = config['network']['use_3D_features']
    num_joints = len(joint_list)
    
    ###############################################
    # Metrics
    mpjpe_threshold = np.arange(25, 200, 25)
    global_acum_err = 0
    n_data = 0
    correct_poses = [0.] * len(mpjpe_threshold)
    TP = []
    FP = []
    for i in range(len(mpjpe_threshold)):
        TP.append([])
        FP.append([])
    n_poses = 0
    n_gt = 0
    # n_matching_poses = 0
    time_inference = 0.
    time_inference_per_person = 0.
    DATASTEP = config['visualization']['data_step']
    ###############################################
    
    # Load dataset
    dataset = HPE_Dataset(config)
    
    total_data = 0
    n_input = 0
    batch_size = 1
    
    loader   = DataLoader(dataset, batch_size=batch_size, collate_fn=hpe_supervised_collate_fn)
    
    time_ini = time.time()
    
    with torch.no_grad():
        for graph, gt in tqdm(loader):
            graph = graph.to(device)
            gt = gt.to(device)
            estimation = model(graph)
            estimation = estimation.squeeze().reshape((-1,3))
            gt = gt.squeeze().reshape((-1,3))
            
            err_per_joint = torch.nn.functional.pairwise_distance(estimation, gt, p=2)
            err = torch.mean(err_per_joint).item()*10
            global_acum_err += err
            n_poses += 1
            
            # if n_poses>1000:
            #     break
            
            for i_th, th in enumerate(mpjpe_threshold):
                if err < th:  # In mm
                    correct_poses[i_th] += 1
                    TP[i_th].append(1)
                    FP[i_th].append(0)
                else:
                    TP[i_th].append(0)
                    FP[i_th].append(1)

    time_inference = time.time()-time_ini                    
            
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    for i_th, th in enumerate(mpjpe_threshold):
        TP_np = np.array(TP[i_th])
        FP_np = np.array(FP[i_th])
        TP_np = np.cumsum(TP_np)
        FP_np = np.cumsum(FP_np)
        recall = TP_np / (n_poses + 1e-5)
        precise = TP_np / (TP_np + FP_np + 1e-5)
        total_num = len(TP[i_th])
        
        for n in range(total_num - 2, -1, -1):
            precise[n] = max(precise[n], precise[n + 1])
        
        precise = np.concatenate(([0], precise, [0]))
        recall = np.concatenate(([0], recall, [1]))
        index = np.where(recall[1:] != recall[:-1])[0]
        ap = np.sum((recall[index + 1] - recall[index]) * precise[index + 1])
        
        print(f'Threshold {th:3d}mm - AP: {ap:.4f}, Precision: {precise[-2]:.4f}, Recall: {recall[-2]:.4f}')
    
    print("\n" + "-"*60)
    if n_poses > 0:
        mpjpe = global_acum_err / n_poses
        print(f"Mean MPJPE: {mpjpe:.2f} mm")
        print(f'\nMean inference time per frame: {time_inference / n_poses:.4f} s')
    else:
        print("No matching poses found")
    
    print(f'Total predicted poses: {n_poses}')
    print("="*60)
            
