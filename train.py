import os
import sys
import yaml
import torch
import argparse
import numpy as np
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from utils.models import GATNetwork, RGCNetwork, GCNetwork
from utils.training_utils import calc_error, calc_error_virtual_and_real
from dataset.custom_dataset import HPE_Dataset, hpe_collate_fn, hpe_VR_collate_fn


def train(model, train_loader, val_loader, extr_mat, intr_mat, dist_coefs, optimizer, savepath, val_interval, patience, scheduler=None, device=None, epochs=10):
    """
    Args:
        model:         GATNetwork instance
        train_loader:  DataLoader con collate_fn=hpe_collate_fn
        val_loader:    DataLoader con collate_fn=hpe_collate_fn
        optimizer:     e.g. optim.Adam(model.parameters(), lr=1e-3)
        scheduler:     opcional
        device:        torch.device
        epochs:        número de épocas

    Returns:
        history (dict): {'train_loss': [...], 'val_loss': [...]}
    """

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):

        # Training
        model.train()
        total_loss, total = 0.0, 0

        for batch_graph, batch_original2D, batch_extr, batch_intr, batch_distCoef, batch_NCams in train_loader:
        # for batch_graph, batch_original2D in train_loader:            
            batch_graph = batch_graph.to(device)
            batch_size  = batch_graph.num_graphs
            batch_original2D = batch_original2D.to(device)
            batch_extr = batch_extr.to(device)
            batch_intr = batch_intr.to(device)
            batch_distCoef = batch_distCoef.to(device)
            batch_NCams = batch_NCams.to(device)

            # Forward pass
            outputs = model(batch_graph)  # [batch_size, 3 * num_joints] (coordinates in cm)
            outputs = outputs  # coordinates in mm
            # print(outputs)

            # Calculate projection error
            # error2D = calc_error(outputs, batch_original2D, extr_mat, intr_mat, dist_coefs, device)
            error2D = calc_error_virtual_and_real(outputs, batch_original2D, batch_extr, batch_intr, batch_distCoef, batch_NCams, device)
            loss = error2D.mean()

            # if torch.isnan(error2D).any():
            #     torch.set_printoptions(profile="full")
            #     print(batch_graph.x)
            #     exit()

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=2, norm_type=2.0)
            optimizer.step()

            total_loss += loss.detach().item() * batch_size
            total      += batch_size

        train_loss = total_loss / total

        # Evaluation
        model.eval()
        val_loss, val_total = 0.0, 0

        with torch.no_grad():
            for batch_graph, batch_original2D, batch_extr, batch_intr, batch_distCoef, batch_NCams in val_loader:
            # for batch_graph, batch_original2D in val_loader:
                batch_graph = batch_graph.to(device)
                batch_size  = batch_graph.num_graphs
                batch_original2D = batch_original2D.to(device)
                batch_extr = batch_extr.to(device)
                batch_intr = batch_intr.to(device)
                batch_distCoef = batch_distCoef.to(device)
                batch_NCams = batch_NCams.to(device)

                # Forward pass
                outputs = model(batch_graph)  # [batch_size, 3 * num_joints] (coordinates in cm)
                outputs = outputs  # coordinates in mm
                
                # Calculate projection error
                # error2D = calc_error(outputs, batch_original2D, extr_mat, intr_mat, dist_coefs, device)
                error2D = calc_error_virtual_and_real(outputs, batch_original2D, batch_extr, batch_intr, batch_distCoef, batch_NCams, device)
                loss    = error2D.mean()

                val_loss  += loss.item() * batch_size
                val_total += batch_size

        val_loss /= val_total
        
        # Check val_loss each VALIDATION_INTERVAL epochs 
        if epoch % val_interval == 0:    
            if val_loss < best_val_loss:
                # Update model.pth file
                best_val_loss = val_loss
                
                torch.save(create_chekpoint(model), savepath)
                print(f"Model saved to {savepath}")
                
                # Early stopping
                patience_counter = 0
                
            else:
                # Early stopping
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping: Model loss has not improved for {patience*val_interval} epochs (Current best: {best_val_loss})")
                    break

        if scheduler is not None:
            scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return history


def create_chekpoint(model):
    config = {
        'model_type':      model.model_type,
        'in_channels':     model.in_channels,
        'out_channels':    model.out_channels,
        'hidden_channels': model.hidden_channels,
    }

    specific_params = {
        "GATNetwork": ['heads', 'concat'],
        "RGCNetwork": ['num_relations', 'num_bases'],
        "GCNetwork":  ['improved']
    }
    
    if model.model_type not in specific_params:
        raise ValueError(f"Error: Model type '{model.model_type}' not recognized for saving.")

    fields = specific_params[model.model_type]
    for field in fields:
        config[field] = getattr(model, field)

    checkpoint = {
        'config': config,
        'state': model.state_dict()
    }

    return checkpoint

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train GATNetwork for Human Pose Estimation')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file containing the command argumnets')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    if not config['execution']['training'] and not config['testing']['test_cam_list']:
        raise ValueError("test_cam_lis must have value if dataset is used in testing")
    

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = HPE_Dataset(config)
    print(f"Dataset size: {len(dataset)} frames")
    
    
    epochs       = config['training_params']['epochs']
    batch_size   = config['training_params']['batch_size']
    lr           = config['training_params']['lr']
    val_split    = config['training_params']['val_split']
    val_interval = config['training_params']['validation_interval']
    patience     = config['training_params']['patience']

    val_size   = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Train: {train_size} | Val: {val_size}")

    extr_mat = None #dataset.real_cams_extr_mat.to(device)
    intr_mat = None #dataset.real_cams_intr_mat.to(device)
    dist_coefs = None #dataset.real_cams_dist_coefs.to(device)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers = 4, shuffle=True,  collate_fn=hpe_VR_collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, num_workers = 4, shuffle=False, collate_fn=hpe_VR_collate_fn)


    num_joints      = len(dataset.joint_list)
    in_channels     = len(dataset.all_features)  
    print("in channels", in_channels) 
    out_channels    = 3 * num_joints
    
    star_topology   = config['network']['star_topology']
    model_pth       = config['network']['model_pth']
    hidden_channels = config['network']['hidden_channels']
    model_type      = config['network']['model_type']
    
    if model_type == "GATNetwork":
        model = GATNetwork(in_channels, out_channels, config['network'], star_topology).to(device)
    elif model_type == "RGCNetwork":
        model = RGCNetwork(in_channels, out_channels, config['network'], star_topology).to(device)
    elif model_type == "GCNetwork":
        model = GCNetwork(in_channels, out_channels, config['network'], star_topology).to(device)
    else:
        raise ValueError(f"Error: Model type '{model_type}' not recognized for saving.")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


    optimizer = optim.Adam(model.parameters(), lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    scheduler = None

    history = train(
        model       = model,
        train_loader= train_loader,
        val_loader  = val_loader,
        extr_mat    = extr_mat,
        intr_mat    = intr_mat, 
        dist_coefs  = dist_coefs,
        optimizer   = optimizer,
        savepath    = model_pth,
        val_interval= val_interval,
        patience    = patience,   
        scheduler   = scheduler,
        device      = device,
        epochs      = epochs
    )

    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(history['train_loss'], label='Train loss')
        plt.plot(history['val_loss'],   label='Val loss')
        plt.xlabel('Epoch')
        plt.ylabel('Reprojection error (px)')
        plt.title('Training history')
        plt.legend()
        plt.tight_layout()
        plot_path = str(Path(model_pth).parent / 'training_history.png')
        plt.savefig(plot_path)
        print(f"Loss plot saved to {plot_path}")
    except ImportError:
        pass