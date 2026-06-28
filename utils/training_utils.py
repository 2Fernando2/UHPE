import sys
import yaml
import torch
from torch.utils.data import DataLoader
import argparse
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
dataset_path = ROOT / "utils"
sys.path.append(str(ROOT))
from dataset.custom_dataset import HPE_Dataset, hpe_collate_fn, hpe_VR_collate_fn
import utils.geometric_utils as gu
import dataset.dataset_utils as du
import time

def calc_error(predictions, original2Ds, device):
    """Calculate 2D reprojection error between model predictions and ground truth.

    Args:
        predictions (Tensor): Output of the forward pass. Shape: [batch_size, 3 * num_joints]
                              Coordinates are in the reference camera's coordinate system.
        original2Ds (list):   List of metadata dicts returned by HPE_Dataset.__getitem__.
        device:               torch.device

    Returns:
        error2D (Tensor): Per-sample summed reprojection error. Shape: [batch_size]
    """

    errors = []
    total_valid = 0

    for i, original2D in enumerate(original2Ds):
        L = predictions.shape[1] // 3

        # print(predictions.shape)

        # Extract estimated3D from model output
        #   predictions[i] shape: (3 * L,)
        estimated3D = predictions[i].reshape((L, 3)).transpose(0,1).to(torch.float32) # (3, L)

        # Camera matrices
        id_cams = original2D["id_cams"]
        extr_mat = torch.stack([torch.tensor(original2D["real_cam_params"]['extr'][cam],       dtype=torch.float32) for cam in id_cams]).to(device)  # (N, 3, 4)
        intr_mat = torch.stack([torch.tensor(original2D["real_cam_params"]['intr'][cam],       dtype=torch.float32) for cam in id_cams]).to(device)  # (N, 3, 3)
        dist_coefs = torch.stack([torch.tensor(original2D["real_cam_params"]['distCoef'][cam], dtype=torch.float32) for cam in id_cams]).to(device)  # (N, 5)

        # Transform predicted 3D coords from reference camera RS to world RS
        # estimated3D_world = gu.points_from_FR1_to_FR2_torch(
        #     original2D["extr_world"].to(device), estimated3D) # (3, L) in world RS
        estimated3D_world = estimated3D
        

        # Project world-RS points onto each camera
        projected, correct3D = gu.project3D_torch(extr_mat, intr_mat, dist_coefs, estimated3D_world)  # (N, 2, L)
        projected = projected.to(device)
        correct3D = correct3D.to(device)
        
        # print('---------------------- Projected -------------------------')
        # print(torch.mean(projected))
        # if torch.isnan(projected).any():
        #     print('NAN in Projection')
        #     exit()

        # if torch.isinf(projected).any():
        #     print('INF in Projection')
        #     exit()

        # Calculate error 
        #   only compare common cameras
        available_cams = original2D["available_cams"]
        coords2D  = original2D["coord2D"].to(device)  # (num_available_cams, 3, L)
        gt_coords = coords2D[:, 0:2, :]               # (num_available_cams, 2, L)
        visible   = coords2D[:, 2, :]                 # (num_available_cams, L)

        common_cams  = [cam for cam in id_cams if cam in available_cams]
        proj_indices = [list(id_cams).index(cam)       for cam in common_cams]
        gt_indices   = [list(available_cams).index(cam) for cam in common_cams]

        proj_common = projected[proj_indices]    # (num_common, 2, L)
        correct3D_common = correct3D[proj_indices]
        gt_common   = gt_coords[gt_indices]      # (num_common, 2, L)
        vis_common  = visible[gt_indices]        # (num_common, L)

        valid_mask = vis_common > 0.5          # (num_common, L)
        # valid_mask = torch.logical_and(valid_mask, original2D['available_3D']>0.5)
        # Euclidean distance
        diff         = proj_common - gt_common                        # (num_common, 2, L)
        # print(diff[:,0,:][valid_mask])
        error_matrix = torch.sum(torch.abs(diff), dim=1) #torch.norm(diff,p=2, dim=1)                      # (num_common, L)
        sample_error = error_matrix[valid_mask].sum()                    # escalar con grad_fn

        num_valid     = valid_mask.sum().item()
        total_valid  += num_valid
        errors.append(sample_error/max(num_valid, 1))
        # print(f"Sample {i} - mean error: {errors[i].item():.4f} px")
    
    error2D = torch.stack(errors) #[batch_size] with grad
    total_error = error2D.sum().item()
    # print('total valid', total_valid)
    # print(f"Batch  - mean error: {total_error / max(len(original2Ds), 1):.4f} px | total error: {total_error:.4f}")
    if torch.isinf(error2D).any():
        print('INF in error2D')
    
    return error2D


def calc_error(predictions, coords2D, extr_mat, intr_mat, dist_coefs, device):
    """Calculate 2D reprojection error between model predictions and ground truth.

    Args:
        predictions (Tensor): Output of the forward pass. Shape: [batch_size, 3 * num_joints]
                              Coordinates are in the reference camera's coordinate system.
        original2Ds (list):   List of metadata dicts returned by HPE_Dataset.__getitem__.
        device:               torch.device

    Returns:
        error2D (Tensor): Per-sample summed reprojection error. Shape: [batch_size]
    """

    estimated3D = predictions.reshape((-1, 3)).transpose(0,1).to(torch.float32) # (3, L)
    projected, correct3D = gu.project3D_torch(extr_mat, intr_mat, dist_coefs, estimated3D)  # (N, 2, L)
    
    gt_coords = coords2D[:, 0:2, :]               # (num_available_cams, 2, L)
    visible   = coords2D[:, 2, :]                 # (num_available_cams, L)

    # valid_mask = torch.logical_and(visible > 0.5, correct3D > 0.5)          # (num_common, L)
    valid_mask = visible > 0.5          # (num_common, L)    

    diff = (projected - gt_coords)                        # (num_common, 2, L)
    
    # valid_mask = torch.logical_and(visible > 0.5, torch.isfinite(diff[:, 0, :]))          # (num_common, L)
    # valid_mask = torch.logical_and(valid_mask, torch.isfinite(diff[:, 1, :]))

    diff = torch.clamp(diff, min=-500., max=500.)    
    diff = diff/1000.

    
    error_matrix = torch.sum(torch.abs(diff), dim=1) #torch.norm(diff,p=2, dim=1)                      # (num_common, L)
    sample_error = error_matrix[valid_mask].sum()                    # escalar con grad_fn
    
    # if torch.isnan(error_matrix[valid_mask]).any():
    #     torch.set_printoptions(profile="full")
    #     print('NAN in error2D')        
    #     print("NUM VALID", valid_mask.sum())
    #     print("--------------------ERROR MATRIX---------------")
    #     print(error_matrix[valid_mask])
    #     print("--------------------X GT-----------------------")
    #     print(gt_coords[:, 0, :][valid_mask])
    #     print("--------------------Y GT-----------------------")
    #     print(gt_coords[:, 1, :][valid_mask])        
    #     print("--------------------X PROJECTED-----------------------")
    #     print(projected[:, 0, :][valid_mask])
    #     print("--------------------Y PROJECTED-----------------------")
    #     print(projected[:, 1, :][valid_mask])        



    num_valid     = max(valid_mask.sum().item(), 1)
    error = sample_error / num_valid

    return error

def calc_error_virtual_and_real(predictions, coords2D, extr_mat, intr_mat, dist_coefs, nCams, device):
    """Calculate 2D reprojection error between model predictions and ground truth.

    Args:
        predictions (Tensor): Output of the forward pass. Shape: [batch_size, 3 * num_joints]
                              Coordinates are in the reference camera's coordinate system.
        original2Ds (list):   List of metadata dicts returned by HPE_Dataset.__getitem__.
        device:               torch.device

    Returns:
        error2D (Tensor): Per-sample summed reprojection error. Shape: [batch_size]
    """
    batch_size = predictions.shape[0]
    estimated3D = predictions.reshape((batch_size, -1, 3)).transpose(1,2).to(torch.float32) # (B, 3, L)

    projected, correct3D = gu.project_batch_3D_torch(extr_mat, intr_mat, dist_coefs, estimated3D, nCams)  # (N, 2, L)
    
    gt_coords = coords2D[:, 0:2, :]               # (num_available_cams, 2, L)
    visible   = coords2D[:, 2, :]                 # (num_available_cams, L)

    # valid_mask = torch.logical_and(visible > 0.5, correct3D > 0.5)          # (num_common, L)
    valid_mask = visible > 0.5          # (num_common, L)    

    diff = (projected - gt_coords)                        # (num_common, 2, L)
    
    diff = torch.clamp(diff, min=-500., max=500.)    
    diff = diff/1000.

    
    error_matrix = torch.sum(torch.abs(diff), dim=1)
    sample_error = error_matrix[valid_mask].sum()
    
    num_valid     = max(valid_mask.sum().item(), 1)
    error = sample_error / num_valid

    return error

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using CUDA')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    # torch.set_printoptions(profile='full')

    parser = argparse.ArgumentParser(description='Test calc_error')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file containing the command argumnets')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    if not config['execution']['training'] and not config['testing']['test_cam_list']:
        raise ValueError("test_cam_lis must have value if dataset is used in testing")
    
    dataset = HPE_Dataset(config)

    extr_mat = dataset.real_cams_extr_mat.to(device)
    intr_mat = dataset.real_cams_intr_mat.to(device)
    dist_coefs = dataset.real_cams_dist_coefs.to(device)

    batch_size = config['training_params']['batch_size']
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,  collate_fn=hpe_VR_collate_fn)
    
    use_3D_features = config['network']['use_3D_features']
    joint_list      = config['network']['joint_list']
    all_features = du.get_all_features(joint_list, use_3D_features)
    norm_vector = torch.Tensor(du.get_norm_vector(joint_list, use_3D_features))
    for batch_graph, batch_original2D, batch_extr, batch_intr, batch_distCoef, batch_NCams in data_loader:
    # for batch_graph, original2D in data_loader:
        
        predictions = []
        available_3D_all = []
        current_batch_size = batch_graph.num_graphs
        for b in range(current_batch_size):
            idx = batch_graph.ptr[b].item()

            feats = batch_graph.x[idx] * norm_vector
            cur_3D = torch.cat([feats[[all_features.index(str(j) + '_x3d'),
                    all_features.index(str(j) + '_y3d'),
                    all_features.index(str(j) + '_z3d')]]
                    for j in joint_list])  
            predictions.append(cur_3D)
            # available_3D = [feats[all_features.index(str(j) + '_estimated')] for j in joint_list]
            # available_3D = torch.stack(available_3D).repeat((5, 1)).to(device)
            # available_3D_all.append(available_3D)

            
        predictions = torch.stack(predictions).to(device)
        # available_3D_all = torch.cat(available_3D_all, dim=1).to(device)                

        time_before = time.time()
        # error = calc_error(predictions, original2D.to(device), extr_mat, intr_mat, dist_coefs, device)
        nCams = batch_NCams.to(device)
        batch_extr_mat = batch_extr.to(device)
        batch_intr_mat = batch_intr.to(device)
        batch_dist_coefs = batch_distCoef.to(device)


        # nCams = torch.tensor([5]*current_batch_size).to(device)
        # batch_extr_mat = extr_mat.repeat(current_batch_size, 1, 1)
        # batch_intr_mat = intr_mat.repeat(current_batch_size, 1, 1)
        # batch_dist_coefs = dist_coefs.repeat(current_batch_size, 1)
        error = calc_error_virtual_and_real(predictions, batch_original2D.to(device), batch_extr_mat, batch_intr_mat, batch_dist_coefs, nCams, device)

        time_after = time.time()
        print(f"Batch  - mean error: {error}, time {time_after-time_before}")

