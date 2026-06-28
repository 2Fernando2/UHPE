from statistics import median
import sys
import torch
import numpy as np
import itertools
import cv2


def get_parameters(cam_parameters):
    cam = {}
    cam['resolution'] = np.array(cam_parameters['resolution'])
    cam['K'] = np.array(cam_parameters['K']).reshape((3,3))
    cam['distCoef'] = np.array(cam_parameters['distCoef'])
    cam['R'] = np.array(cam_parameters['R']).reshape((3,3))
    cam['t'] = np.array(cam_parameters['t']).reshape((3,1)) #3D in meters
    cam['ext'] = np.zeros((3,4), dtype=np.float64)
    cam['ext'][:,:-1] = cam['R'][:]
    cam['ext'][:,-1] = cam['t'][:,0]
    cam['proj'] = np.matmul(cam['K'], cam['ext'])
    return cam

def TR_from_camera_to_world(Tr):
    """ Transform the extrinsic matrix Tr from the camera
    frame of reference to the world frame of reference.
    Return the new transformation matrix.
    """

    newR = Tr[:,:-1].transpose()
    newT = -newR@Tr[:,-1]
    newTR = np.concatenate((newR, newT[:, np.newaxis]), axis=1)

    return newTR

def TR_from_camera_to_camera(camsrc_Tr, camdst_Tr):
    """ Transform the extrinsic matrix Tr from the world
    frame of reference to a camera frame of reference (cam_Tr).
    Return the new extrinsic matrix: (R2*R1^T, -(R2*R1^T)*T1+T2
    """

    R1T = camsrc_Tr[:,:-1].transpose()
    T1 = camsrc_Tr[:,-1]
    R2 = camdst_Tr[:,:-1]
    T2 = camdst_Tr[:,-1]
    newR = R2@R1T
    newT = -newR@T1 + T2
    newTR = np.concatenate((newR, newT[:, np.newaxis]), axis=1)

    return newTR


def points_from_FR1_to_FR2_torch(Tr, X):
    """ Transform the 3D points X from one
    frame of reference (FR1) to the another one (FR2).
    Return the new point list.
    Tr (torch): transformation matrix from FR1 to FR2 (3x4)
    X (torch): torch array with shape (3xL), with L the number of
       points
    For transforming points from world to camera, use the extrinsic
    matrix of the camera as Tr.
    """

    Xh = torch.cat((X, torch.ones((1, X.shape[-1]), dtype=X.dtype, device=X.device)), dim=0)
    Xc = torch.matmul(Tr, Xh) # (3xL)

    return Xc

def points_from_FR1_to_FR2(Tr, X):
    """ Transform the 3D points X from one
    frame of reference (FR1) to the another one (FR2).
    Return the new point list.
    Tr: transformation matrix from FR1 to FR2 (3x4)
    X: numpy array with shape (3xL), with L the number of
       points
    For transforming points from world to camera, use the extrinsic
    matrix of the camera as Tr.
    """

    Xh = np.concatenate((X, np.ones((1, X.shape[-1]), dtype=np.float32)), axis=0)
    Xc = Tr@Xh # (3xL)

    return Xc


# transform homogeneous into standard coordinates (2 components)
def from_homogeneous(v): 
    return (v/v[-1])[:-1]

# transform homogeneous into standard coordinates (3 components)
def from_homogeneous2(v):
    return (v/v[-1])


def apply_fisheye_distortion_torch(kd, v):
    v2 = v.clone()
    r = torch.norm(v[:-1][:], dim=0)
    theta = torch.atan(r)
    theta2 = theta*theta
    theta4 = theta2*theta2
    theta6 = theta4*theta2
    theta8 = theta4*theta4
    theta_d = theta*(1 + kd[0]*theta2 + kd[1]*theta4 + kd[2]*theta6 + kd[3]*theta8)
    correction = torch.div(theta_d, r)
    v2[0][:] = v[0][:]*correction
    v2[1][:] = v[1][:]*correction
    return v2


def apply_distortion_torch(kd, v):
    v2 = v.clone()
    r = torch.norm(v[:-1][:], dim=0)
    r = r*r
    v2[0][:] = v[0][:]*(1 + kd[0]*r + kd[1]*r*r + kd[4]*r*r*r) + 2*kd[2]*v[0][:]*v[1][:] + kd[3]*(r + 2*v[0][:]*v[0][:])
    v2[1][:] = v[1][:]*(1 + kd[0]*r + kd[1]*r*r + kd[4]*r*r*r) + 2*kd[3]*v[0][:]*v[1][:] + kd[2]*(r + 2*v[1][:]*v[1][:])


    return v2

def apply_distortion_numpy(kd, v):
    v2 = v.copy()
    r = np.linalg.norm(v[:-1,:], dim=0)
    r = r*r
    v2[0,:] = v[0,:]*(1 + kd[0]*r + kd[1]*r*r + kd[4]*r*r*r) + 2*kd[2]*v[0,:]*v[1,:] + kd[3]*(r + 2*v[0,:]*v[0,:])
    v2[1,:] = v[1,:]*(1 + kd[0]*r + kd[1]*r*r + kd[4]*r*r*r) + 2*kd[3]*v[0,:]*v[1,:] + kd[2]*(r + 2*v[1,:]*v[1,:])

    return v2

def project3D_torch(Me, Mi, kd, X):
    """ Projects L points X (3xL) into N cameras,
    Me (torch): extrinsic matrices with shape [N, 3, 4].
    Mi (torch) : intrinsic matrices with shape [N, 3, 3].
    kd (torch): distortion coefficients with shape [N, 5]
    result (torch): projected points with shape [N, 2, L]
    """
    Xh = torch.cat((X, torch.ones((1, X.shape[-1]), dtype=X.dtype, device=X.device)), dim=0)
    x = torch.matmul(Me, Xh) # (Nx3xL)

    # Normalize by Z coordinate    
    zn = x[:, 2:3, :] # (N, 1, L)
    xn = x[:, 0:2, :] / zn # (N, 2, L)
    
    # Calculate distorsion
    x_loc = xn[:, 0, :]
    y_loc = xn[:, 1, :]
    
    r2 = x_loc**2 + y_loc**2
    r4 = r2**2
    r6 = r2**3
    
    k = kd.unsqueeze(-1) 
    r = (1 + k[:,0]*r2 + k[:,1]*r4 + k[:,4]*r6)
    
    # Apply radial distorsion + tangencial 
    xd_x = x_loc * r + 2*k[:,2]*x_loc*y_loc + k[:,3]*(r2 + 2*x_loc**2)
    xd_y = y_loc * r + 2*k[:,3]*x_loc*y_loc + k[:,2]*(r2 + 2*y_loc**2)
    
    xd = torch.stack([xd_x, xd_y, torch.ones_like(xd_x)], dim=1) # (N, 3, L)
    
    result = torch.matmul(Mi, xd)
    result = result[:,0:2,:]
    # result[:, 0, :] = result[:, 0, :] * (zn[:, 0, :]>0.01)
    # result[:, 1, :] = result[:, 1, :] * (zn[:, 0, :]>0.01)
    filter3D = (zn[:, 0, :]>100)
    # result[:, 0, :][filte3D] = 0
    # result[:, 1, :][filte3D] = 0

    
    # print((zn[:, 0, :]>100).shape)
    # nan_indices = torch.isnan(result).nonzero()
    # zzero_indices = (zn[:,0, :]<=0.01).nonzero()
    # print(nan_indices.tolist())
    # print('--------------')
    # print(zzero_indices.tolist())
    # print(result[nan_indices])
    return result, filter3D

def project_batch_3D_torch(Me, Mi, kd, X, NCams):
    """ Projects a batch with BxL 3D points into a batch of N1, N2, ..., NB cameras: 
    N = N1+N2+...+NB
    X (torch): 3D points with shape(Bx3xL),
    Me (torch): extrinsic matrices with shape [N, 3, 4].
    Mi (torch) : intrinsic matrices with shape [N, 3, 3].
    kd (torch): distortion coefficients with shape [N, 5]
    NCams (torch): tensor cointaining the number of cameras of each batch sample [1, B]
    result (torch): projected points with shape [N, 2, L]
    """
    Xh = torch.cat((X, torch.ones((X.shape[0], 1, X.shape[-1]), dtype=X.dtype, device=X.device)), dim=1) #(Bx4xL)
    Xh_rep = torch.repeat_interleave(Xh, NCams, dim=0) #(Nx4xL)
     
    x = torch.bmm(Me, Xh_rep) # (Nx3xL)

    # Normalize by Z coordinate    
    zn = x[:, 2:3, :] # (N, 1, L)
    xn = x[:, 0:2, :] / zn # (N, 2, L)
    
    # Calculate distorsion
    x_loc = xn[:, 0, :]
    y_loc = xn[:, 1, :]
    
    r2 = x_loc**2 + y_loc**2
    r4 = r2**2
    r6 = r2**3
    
    k = kd.unsqueeze(-1) 
    r = (1 + k[:,0]*r2 + k[:,1]*r4 + k[:,4]*r6)
    
    # Apply radial distorsion + tangencial 
    xd_x = x_loc * r + 2*k[:,2]*x_loc*y_loc + k[:,3]*(r2 + 2*x_loc**2)
    xd_y = y_loc * r + 2*k[:,3]*x_loc*y_loc + k[:,2]*(r2 + 2*y_loc**2)
    
    xd = torch.stack([xd_x, xd_y, torch.ones_like(xd_x)], dim=1) # (N, 3, L)
    
    result = torch.bmm(Mi, xd)
    result = result[:,0:2,:]
    
    filter3D = (zn[:, 0, :]>100)
    
    return result, filter3D


def project3D_numpy(Me, Mi, kd, X):
    """ Projects L points X (3xL) into N cameras,
    Me: extrinsic matrices with shape [N, 3, 4].
    Mi: intrinsic matrices with shape [N, 3, 3].
    kd: distortion coefficients with shape [N, 5]
    result: projected points with shape [N, 2, L]
    """
    Xh = np.concatenate((X, np.ones((1, X.shape[-1]), dtype=np.float32)), axis=0)
    # x = Me[:,:,0:3]@X + Me[:,:,3] # (Nx3xL)
    x = Me@Xh # (Nx3xL)

    # filte3D = (x[:, 2, :]>10)
    x[:,0,:] = x[:,0,:]/x[:,2,:]
    x[:,1,:] = x[:,1,:]/x[:,2,:]
    x[:,2,:] = x[:,2,:]/x[:,2,:]
    r = np.linalg.norm(x[:, :-1, :], axis=1)
    r = r*r
    xd = x.copy()
    kd_e = kd[..., np.newaxis]
    # print(r.shape, kd[:,0].shape, (kd[:,0]*r).shape)
    xd[:,0,:] = x[:,0,:]*(1 + kd_e[:,0,:]*r + kd_e[:,1,:]*r*r + kd_e[:,4,:]*r*r*r) + 2*kd_e[:,2,:]*x[:,0,:]*x[:,1,:] + kd_e[:,3,:]*(r + 2*x[:,0,:]*x[:,0,:])
    xd[:,1,:] = x[:,1,:]*(1 + kd_e[:,0,:]*r + kd_e[:,1,:]*r*r + kd_e[:,4,:]*r*r*r) + 2*kd_e[:,3,:]*x[:,0,:]*x[:,1,:] + kd_e[:,2,:]*(r + 2*x[:,1,:]*x[:,1,:])
    result = Mi@xd
    return result[:,0:2,:]


def triangulate(points_2D, joint_list, camera_matrices, distortion_coefficients, projection_matrices, median_chek_axis, fisheye=None):
    result3D = dict()
    for idx_i in joint_list:
        idx = str(idx_i)
        point3d_list = []
        if idx in points_2D.keys() and len(points_2D[idx]) > 1:
            cam_combinations = itertools.combinations(range(len(points_2D[idx].keys())), 2)
            for comb in cam_combinations:
                cam1 = list(points_2D[idx].keys())[comb[0]]
                cam2 = list(points_2D[idx].keys())[comb[1]]
                point1 = np.array(points_2D[idx][cam1])
                if fisheye is not None and fisheye[cam1]:
                    new_point1 = cv2.fisheye.undistortPoints(np.array([point1]), camera_matrices[cam1], distortion_coefficients[cam1])
                else:
                    new_point1 = cv2.undistortPoints(np.array([point1]), camera_matrices[cam1], distortion_coefficients[cam1])
                point2 = np.array(points_2D[idx][cam2])                                    
                if fisheye is not None and fisheye[cam2]:
                    new_point2 = cv2.fisheye.undistortPoints(np.array([point2]), camera_matrices[cam2], distortion_coefficients[cam2])
                else:
                    new_point2 = cv2.undistortPoints(np.array([point2]), camera_matrices[cam2], distortion_coefficients[cam2])
                point3d = cv2.triangulatePoints(projection_matrices[cam1], projection_matrices[cam2], new_point1, new_point2)
                point3d = point3d[0:3]/point3d[3]
                point3d_list.append(point3d)
            point3d_list = np.array(point3d_list)
            dist_to_0 = point3d_list[:,median_chek_axis]
            median = np.sort(dist_to_0, axis=0)[dist_to_0.shape[0]//2]
            dist_to_median = np.linalg.norm(dist_to_0-median, axis=1)
            new_point3d_list = [point3d_list[i,:] for i in range(dist_to_median.shape[0]) if dist_to_median[i] < 0.05]
            result3D[idx] = np.mean(np.array(new_point3d_list), axis=0)
    return result3D
