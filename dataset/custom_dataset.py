import os
import re
import sys
import json
import torch
import timeit
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch_geometric.data import Data, Dataset, Batch


ROOT = Path(__file__).resolve().parent.parent
dataset_path = ROOT / "utils"
sys.path.append(str(ROOT))
import dataset.dataset_utils as du
import utils.geometric_utils as gu

class HPE_Data(Data):
    def __init__(self, frame, points_3D, cam_params, all_features, camera_names, num_joints, norm_vector, use_3D_features, star_topology, apply_noise=True):
        """Initialize object

        Args:
            frame (dict): Dictionary with each camera information
                Note: This dictionary should be one element of the list in the .json file.
            
            points_3D (dict): Dictionay with each joint 3D position
                Note: Contains only the points where estimation was successful. 
            
            cam_params (dict): Dictionary with camera parameters with the structure:
                - 'K' (np.matrix): Camera intrinsic matrix (3x3)
                - 'distCoef' (np.array): Distortion coefficients
                - 'ext' (np.array): Extrinsic matrix [R|t] (3x4). 
                    Note: With respect to a specific camera's frame of reference
                Note: This dictionary should be the result of get_parameters().
                
            all_features (list): List with all camera features for an specific frame with the structure:
                - 'jointName_x2d (float)': The horizontal coordinate (x-axis) in 2D space
                - 'jointName_y2d' (float): The vertical coordinate (y-axis) in 2D space
                - 'jointName_confidence' (float): Confidence interval (0 to 1) that the joint is correctly detected
                - 'jointName_visibility' (float): Indicate if the joint has been detected by the camera 
                - 'jointName_line_pX (float)': The X-position of the camera center in 3D space
                - 'jointName_line_pY (float)': The Y-position of the camera center in 3D space
                - 'jointName_line_pZ (float)': The Z-position of the camera center in 3D space
                - 'jointName_line_vX (float)': The X-component of the direction vector pointing to the joint
                - 'jointName_line_vY (float)': The Y-component of the direction vector pointing to the joint
                - 'jointName_line_vZ (float)': The Z-component of the direction vector pointing to the joint
                - 'jointName_estimated (float)': Indicates if the node's 3D position was successfully estimated
                - 'jointName_x3d' (float): The horizontal coordinate (x-axis) in 3D space
                - 'jointName_y3d' (float): The vertical coordinate (y-axis) in 3D space
                - 'jointName_z3d' (float): The depth coordinate (z-axis) in 3D space
                - 'intr_matrix (np.array)': Camera intrinsic matrix (3x3)
                - 'extr_matrix (np.array)': Camera extrinsic matrix (3x4)
                - 'disCoef (np.array)': Distorsion coefficients
                
                Note: This list should be the resoult of get_all_features().
                
            camera_names (list): List with all camera names 
                
            num_joints (int): Total number of joints from a camera
                Note: All cameras should have the same number of joints, these may be visible or not. 
            
            norm_vector (list): List where each element is the divisor for the corresponding value in the HPE_Data.x[cam].
            
            use_3D_features(boolean): Indicates whether 3D information will be used as input features
                - True: The node feature vector will include 3D information
                - False: The nodes will only use 2D information 
            
            star_topology (boolean): Indicates whether to use a star topology or a fully connected graph
                - True: The topology is a star type (node 0 connected to all other nodes)
                - False: The topology is a fully connected graph
            
            apply_noise (boolean): Indicates wheter to apply or not noise to the coordinates
                Default: True
            
        """
        
        if frame is None:
            super().__init__()
            return
        
        #Set x.size before super().__init__()
        num_node_features = len(all_features)
        x = torch.zeros((len(camera_names), num_node_features), dtype=torch.float32)
        
        super().__init__(x=x)
        
        # self.frame = frame
        # self.points_3D = points_3D
        # self.cam_params = cam_params
        self.all_features = all_features
        self.camera_names = camera_names
        self.num_joints = num_joints
        self.norm_vector = torch.tensor(norm_vector, dtype=torch.float32)
        self.apply_noise = apply_noise
        
        self.load_camera_data(frame, points_3D, cam_params, use_3D_features, star_topology)
    
    def load_camera_data(self, frame, points_3D, cam_params, use_3D_features, star_topology):
        #Metadata

        self.num_nodes = len(self.camera_names)
        #Star type graph
        self.num_edges = self.num_nodes - 1 
        self.num_node_features = len(self.all_features)
        self.num_edge_features = 0        
        
        
        #Data.x
        self._joint_dictionary = {}
        visibility = {}
        noise = torch.zeros_like(self.x)

        # Gausian noise params
        noise_shift = 5.0  # in pixels
        
        for cam_idx, cam in enumerate(self.camera_names):
            skeleton_list = frame[cam]
            
            #Keypoints (joints)
            joint_dict = skeleton_list[0] #Only one skeleton
            self._joint_dictionary[cam] = joint_dict
            for joint_name, joint in joint_dict.items():
                joint_name_str = str(joint_name)

                if not joint_name in visibility.keys():
                    visibility[joint_name] = 0
                if joint[3]>0.5:
                    visibility[joint_name] += joint[3]

                    start_idx = self.all_features.index(joint_name_str + '_x2d')
                    self.x[cam_idx, start_idx : start_idx+4] = torch.tensor(joint[1:], dtype=torch.float32)
                    
                    noise[cam_idx, start_idx : start_idx+2] = torch.rand(
                        1, 2, dtype=torch.float32
                    ) * noise_shift*2 - noise_shift


        if use_3D_features:    
            for cam_idx, cam in enumerate(self.camera_names):
                skeleton_list = frame[cam]
                
                #Keypoints (joints)
                joint_dict = skeleton_list[0] #Only one skeleton
                for joint_name in joint_dict:
                    joint_name_str = str(joint_name)
                    
                    visible = visibility[joint_name] > 1
                    idx_estimated = self.all_features.index(joint_name_str + '_estimated')
                    if visible and joint_name in points_3D.keys():    
                        coords_3D = points_3D[joint_name].flatten()
                        self.x[cam_idx, idx_estimated : idx_estimated+4] = torch.tensor([1.0, *coords_3D], dtype=torch.float32)
        
        # Apply noise
        if self.apply_noise:
            self.x = self.x + noise
            
        #Cam params
        for cam_idx, cam in enumerate(self.camera_names):
            K_flat = cam_params[cam]['K'].flatten()
            ext_flat = cam_params[cam]['ext'].flatten()
            dist_flat = cam_params[cam]['distCoef'].flatten()
            
            index = self.all_features.index('_intr_matrix_0')
            self.x[cam_idx, index:index+len(K_flat)] = torch.tensor(K_flat, dtype=torch.float)
                
            index = self.all_features.index('_extr_matrix_0')
            self.x[cam_idx, index:index+len(ext_flat)] = torch.tensor(ext_flat, dtype=torch.float)
                
            index = self.all_features.index('_distCoef_0')
            self.x[cam_idx, index:index+len(dist_flat)] = torch.tensor(dist_flat, dtype=torch.float)
    
        
        #Normalize Data.x
        for cam_idx, cam in enumerate(self.camera_names):
            self.x[cam_idx, :] = self.x[cam_idx, :] / self.norm_vector


        # Full conected
        if star_topology:
            # Star topoligy
            self.directed = False 
            list_N = list(range(1, self.num_nodes))
            list_0 = [0 for _ in range(self.num_nodes - 1)]
            list_ALL = list(range(self.num_nodes))
            if self.directed:
                self.edge_index = np.zeros((2, 2*(self.num_nodes-1)+self.num_nodes))
                self.edge_index[0, :] = list_N + list_0 + list_ALL
                self.edge_index[1, :] = list_0 + list_N + list_ALL
            else:
                self.edge_index = np.zeros((2, self.num_nodes-1+self.num_nodes))
                self.edge_index[0, :] = list_N + list_ALL
                self.edge_index[1, :] = list_0 + list_ALL
            self.edge_index
            self.edge_index = torch.tensor(self.edge_index, dtype=torch.long)
        else:    
            adj = torch.ones((self.num_nodes, self.num_nodes))
            self.edge_index = adj.nonzero().t().contiguous()
            self.edge_index = self.edge_index.to(torch.long)


        
        
class HPE_Dataset(Dataset):
    def __init__(self, config):
        """Initialize object

            Args:
                - config (dict): Dictionary containing the Dataset creation attributes 
                    - Note: Sourced from the .yaml configuration file.
        """
        
        self.paramsdir       = config['paths']['paramsdir']
        self.joint_list      = config['network']['joint_list']
        self.star_topology   = config['network']['star_topology']
        self.use_3D_features = config['network']['use_3D_features']
        self.real_cameras    = config['training_params']['real_cameras']
        self.training        = config['execution']['training']
        self.axes_3D         = config['visualization']['axes_3D']
        self.test_cam_list   = config['testing']['test_cam_list']

        trainset     = config['paths']['trainset']
        frames_cache = config['paths']['frames_cache']
        load_frames  = config['execution']['load_frames']
        self.load_cam_data(trainset, load_frames, frames_cache) 

    def get_camera_params(self, json_files):
        self.calib_ids = []
        
        # Extraxt all the json_files paths
        p = Path(json_files)
        root_dir = p.parent
        pattern = p.name
        json_files = list(root_dir.glob(pattern))
        
        for json_file in json_files:
            file_name = os.path.basename(json_file)
            segments = file_name.split('_')
            calib_idx = "_".join(segments[0:2])
            self.calib_ids.append(calib_idx)
        
        camera_params = {}
        self.id_cams = {}

        self.intr_matrices = {}
        self.ext_matrices = {}
        self.distCoefs = {}
        self.resolutions = {}
        self.ext_inv = {}

        self.real_cams_extr_mat = {}
        self.real_cams_intr_mat = {}
        self.real_cams_dist_coefs = {}


        for calib_idx in self.calib_ids:
            cam_idx_list = os.listdir(os.path.join(self.paramsdir, calib_idx))
            self.id_cams[calib_idx] = [f.replace('calib_', '').replace('.json', '') for f in cam_idx_list if f.startswith('calib_')]
            
            camera_params[calib_idx] = {}
            
            self.intr_matrices[calib_idx] = {}
            self.ext_matrices[calib_idx] = {}
            self.distCoefs[calib_idx] = {}
            self.resolutions[calib_idx] = {}
            self.ext_inv[calib_idx] = {}

            calib_dir = os.path.join(self.paramsdir, calib_idx)
            for cam in self.id_cams[calib_idx]:
                params_file = os.path.join(calib_dir, 'calib_'+cam+'.json')
                cam_params = json.load(open(params_file, 'rb'))
                params = gu.get_parameters(cam_params)
                camera_params[calib_idx][cam] = params
                self.intr_matrices[calib_idx][cam] = params['K']
                self.ext_matrices[calib_idx][cam] = params['ext']
                self.distCoefs[calib_idx][cam] = params['distCoef']
                self.resolutions[calib_idx][cam] = params['resolution']
                
                #Precalculate inverted matrices 
                # for transformatrions to differents reference sytems
                R = params['ext'][:, :3]
                t = params['ext'][:, 3:]
                self.ext_inv[calib_idx][cam] = np.hstack((R.T, -R.T @ t))
            
            self.real_cams_extr_mat[calib_idx] = torch.stack([torch.tensor(self.ext_matrices[calib_idx][cam],  dtype=torch.float32) for cam in self.real_cameras])  # (N, 3, 4)
            self.real_cams_intr_mat[calib_idx] = torch.stack([torch.tensor(self.intr_matrices[calib_idx][cam], dtype=torch.float32) for cam in self.real_cameras])  # (N, 3, 3)
            self.real_cams_dist_coefs[calib_idx] = torch.stack([torch.tensor(self.distCoefs[calib_idx][cam], dtype=torch.float32) for cam in self.real_cameras])  # (N, 5)

            
        return camera_params

    def load_cam_data(self, json_files, load_frames, frames_cache_path):
        #Camera params
        self.camera_params = self.get_camera_params(json_files)

        if not load_frames:
            # Extraxt all the json_files paths
            p = Path(json_files)
            root_dir = p.parent
            pattern = p.name
            json_files = list(root_dir.glob(pattern))
            
            #Ensure json_files is a list
            if isinstance(json_files, str):
                json_files = [json_files]        

            #Frame list
            self.frames = []

            for json_file in json_files:
                # print(json_file)
                
                file_name = os.path.basename(json_file)
                segments = file_name.split('_')
                calib_idx = "_".join(segments[0:2])
                print(f"HPE_Dataset: Processing {calib_idx}")
                
                self.input_data = json.load(open(json_file, 'rb'))            

                # nFrames = 0
                for frame_idx, frame in tqdm(enumerate(self.input_data), total=len(self.input_data)):
                    # nFrames+=1
                    # if nFrames > 500:
                    #     break
                    frame_data = {}
                    
                    frame_data['calib_idx'] = calib_idx
                    frame_data['frame_idx'] = frame_idx

                    L = len(self.joint_list)
                    coords2D = torch.zeros(len(self.real_cameras), 3, L, dtype=torch.float32)                    
                    #Calculate the 3D coordenates and cameras with 2D
                    joints_data = {}
                    for cam_name in frame:
                        if cam_name not in self.real_cameras:
                            continue
                        skeleton_list = json.loads(frame[cam_name][0])
                        joints_dict = skeleton_list[0]  #Only one skeleton
                        
                        cam_idx = self.real_cameras.index(cam_name)
                        for joint in joints_dict:
                            if int(joint) not in self.joint_list:
                                continue
                            visibility = joints_dict[joint][3] 
                            if visibility > 0.5:
                                if joint not in joints_data:
                                    joints_data[joint] = {}
                                joints_data[joint][cam_name] = [joints_dict[joint][1], joints_dict[joint][2]]

                                joint_idx = self.joint_list.index(int(joint))
                                coords2D[cam_idx, 0, joint_idx] = joints_dict[joint][1]
                                coords2D[cam_idx, 1, joint_idx] = joints_dict[joint][2]
                                coords2D[cam_idx, 2, joint_idx] = joints_dict[joint][3]

                    frame_data['coords2D'] = coords2D
                       
                    frame_data['3D'] = gu.triangulate(joints_data, self.joint_list, self.intr_matrices[calib_idx],
                                                    self.distCoefs[calib_idx], self.ext_matrices[calib_idx], 
                                                    self.axes_3D['Y'][0])
                    
                    first_cam = list(frame.keys())[0]

                    storedGT = frame[first_cam][3][0]


                    GT = torch.zeros(3*len(self.joint_list), dtype=torch.float32)
                    for j, joint in enumerate(self.joint_list):
                        j_str = str(joint)
                        if j_str in storedGT.keys():
                            GT[j*3:j*3+3] = torch.tensor(storedGT[j_str])

                    frame_data['GT'] = GT

                    #Only append when there is a 3D
                    if frame_data['3D']:
                        # Precalculate 2D projections for all cameras
                        precomputed_joints = du.generate_new_data(frame_data['3D'], self.id_cams[calib_idx], self.camera_params[calib_idx])
                        
                        # Only append the visible joints
                        # joints = {}
                        cams_with_2D = {}
                        for cam in precomputed_joints:
                            # joints[cam] = {}
                            exist2D = 0
                            for joint in precomputed_joints[cam]:
                                data = precomputed_joints[cam][joint]
                                if data[3] > 0.5:
                                    exist2D += 1
                                    # joint_name = data[0]
                                    # joints[cam][joint_name] = data
                            cams_with_2D[cam] = exist2D > 0
                        
                        # frame_data['precomputed_joints'] = joints
                        frame_data['cams_with_2D'] = [cam_name for cam_name, exist2D in cams_with_2D.items() if exist2D]
                        frame_data['available_cams'] = [cam for cam in list(frame.keys()) if cam in frame_data['cams_with_2D']]
                        
                        # #Precalculate coords2D tensor
                        # L = len(self.joint_list)
                        # coords2D = torch.zeros(len(self.real_cameras), 3, L, dtype=torch.float32)
                        # for cam_idx, cam in enumerate(self.real_cameras):
                        #     for joint_idx, joint in enumerate(self.joint_list):
                        #         joint_str = str(joint)
                        #         if joint_str in joints_data and cam in joints_data[joint_str]:
                        #             coords2D[cam_idx, 0, joint_idx] = joints_data[joint_str][cam][0]
                        #             coords2D[cam_idx, 1, joint_idx] = joints_data[joint_str][cam][1]
                        #             coords2D[cam_idx, 2, joint_idx] = 1.0
                        # frame_data['coords2D'] = coords2D

                        self.frames.append(frame_data)
                    
            # Save frames
            torch.save(self.frames, frames_cache_path)

        else:
            # Load frames
            self.frames = torch.load(frames_cache_path, weights_only=False)
        
        #Extract all camera features
        self.all_features = du.get_all_features(self.joint_list, self.use_3D_features)
        
        #Calculate num_joints
        self.num_joints = len(self.joint_list)
        
        # Normalize vector
        self.norm_vector = du.get_norm_vector(self.joint_list, self.use_3D_features)


    def __getitem__(self, frame_idx):  
        # graph, original2D = self.get_item_real_cams(frame_idx)

        if self.training:
            # graph, original2D = self.get_item_supervised(frame_idx)
            graph, original2D = self.get_item_virtual_and_real_cams(frame_idx)
        else:
            graph, original2D = self.get_item_test(frame_idx)

        return graph, original2D
    
    def get_item_test(self, frame_idx):
        cam_list = [cam for cam in self.real_cameras if cam in self.frames[frame_idx]['cams_with_2D']]
        # print(cam_list)
        num_cameras = len(cam_list)
        calib_idx = self.frames[frame_idx]['calib_idx']
        
        points2D = self.frames[frame_idx]['coords2D']
        
        pointsDict = dict()
        for cam in cam_list:
            i = self.real_cameras.index(cam)
            skeleton = dict()
            for j, keypoint in enumerate(self.joint_list):
                x, y, v = points2D[i, 0, j], points2D[i, 1, j], points2D[i, 2, j]
                idx = keypoint
                confidence = v
                skeleton[keypoint] = (idx, x, y, v, confidence)
            pointsDict[cam] = [skeleton]    

        original_3D_dict = self.frames[frame_idx]['3D']

        cam_params = {}
        for cam in cam_list:
            cam_params[cam] = {}
            cam_params[cam]['K'] = self.intr_matrices[calib_idx][cam]
            cam_params[cam]['distCoef'] = self.distCoefs[calib_idx][cam]
            cam_params[cam]['ext'] = self.ext_matrices[calib_idx][cam]
            cam_params[cam]['resolution'] = self.resolutions[calib_idx][cam]

        
        graph = HPE_Data(pointsDict, original_3D_dict, cam_params, self.all_features, cam_list, self.num_joints, self.norm_vector, self.use_3D_features, self.star_topology, apply_noise=False)
        
        original2D = {}

        original2D["coord2D"] = self.frames[frame_idx]['coords2D']
        
        original2D["extr"] = self.real_cams_extr_mat[calib_idx]
        original2D["intr"] = self.real_cams_intr_mat[calib_idx]
        original2D["distCoef"] = self.real_cams_dist_coefs[calib_idx]
        

        # NCams
        NCams = num_cameras
        original2D["NCams"] = torch.tensor(NCams, dtype=torch.int32)
        
        original2D["GT"] = self.frames[frame_idx]['GT']

        return graph, original2D
        
        

    def get_item_real_cams(self, frame_idx):
        if self.training:
            num_cameras = random.randint(2, 7)
            self.cam_list_rand = random.sample(self.frames[frame_idx]['cams_with_2D'], k=num_cameras)   
        else:
            self.cam_list_rand = [cam for cam in self.test_cam_list if cam in self.frames[frame_idx]['cams_with_2D']]

        self.reference_camera = random.choice(self.cam_list_rand) 
        
        calib_idx = self.frames[frame_idx]['calib_idx']
        
        #Transform the coordenates 3D from the original Reference System
        # to the reference_camera system
        Mext_RS = self.camera_params[calib_idx][self.reference_camera]['ext']
        original_3D_dict = self.frames[frame_idx]['3D']
        new_3D_dict = {}
        for joint in original_3D_dict:
            new_3D_dict[joint] = original_3D_dict[joint] #gu.points_from_FR1_to_FR2(Mext_RS, original_3D_dict[joint])
            # new_3D_dict[joint] = gu.points_from_FR1_to_FR2(Mext_RS, original_3D_dict[joint])
        
        #Transform the extrinsic matrix for each camera from the original
        # Reference System to the reference_camera system
        Mext_new_dit = {}
        for cam in self.cam_list_rand:
            Mext_cam = self.camera_params[calib_idx][cam]['ext']
            Mext_new_dit[cam] = Mext_cam #gu.TR_from_camera_to_camera(Mext_RS, Mext_cam)
            # Mext_new_dit[cam] = gu.TR_from_camera_to_camera(Mext_cam, Mext_RS)

        #Create a cam_params dictionary with the new matrices
        new_cam_params = {}
        for cam in self.cam_list_rand:
            new_cam_params[cam] = {}
            new_cam_params[cam]['K'] = self.intr_matrices[calib_idx][cam]
            new_cam_params[cam]['distCoef'] = self.distCoefs[calib_idx][cam]
            new_cam_params[cam]['ext'] = Mext_new_dit[cam]
            new_cam_params[cam]['resolution'] = self.resolutions[calib_idx][cam]

        
        # precomputed_joints = self.frames[frame_idx]['precomputed_joints']
        precomputed_joints = du.generate_new_data(new_3D_dict, self.cam_list_rand, self.camera_params[calib_idx])
        joints_dict = {cam: precomputed_joints[cam] for cam in self.cam_list_rand if cam in precomputed_joints}


        #Create the dictionry with each canera information
        frame = {}
        N = len(self.cam_list_rand)
        L = len(self.joint_list)
        for cam in self.cam_list_rand:
            skeleton_list = []
            skeleton = joints_dict[cam]
            skeleton_list.append(skeleton) #Only one skeleton                
            frame[cam] = skeleton_list
        
        apply_noise = self.training == True

        graph = HPE_Data(frame, new_3D_dict, new_cam_params, self.all_features, self.cam_list_rand, self.num_joints, self.norm_vector, self.use_3D_features, self.star_topology, apply_noise)

        original2D = {}

        original2D["coord2D"] = self.frames[frame_idx]['coords2D']
       
        return graph, original2D

    def get_item_virtual_and_real_cams(self, frame_idx, use_all=True):
        if self.training:
            num_cameras = random.randint(2, 7)
            self.cam_list_rand = random.sample(self.frames[frame_idx]['cams_with_2D'], k=num_cameras)   
        else:
            self.cam_list_rand = [cam for cam in self.test_cam_list if cam in self.frames[frame_idx]['cams_with_2D']]
            num_cameras = len(self.cam_list_rand)
        
        self.reference_camera = random.choice(self.cam_list_rand) 

        calib_idx = self.frames[frame_idx]['calib_idx']
        
        #Transform the coordenates 3D from the original Reference System
        # to the reference_camera system
        Mext_RS = self.camera_params[calib_idx][self.reference_camera]['ext']
        original_3D_dict = self.frames[frame_idx]['3D']
        new_3D_dict = {}
        for joint in original_3D_dict:
            new_3D_dict[joint] = original_3D_dict[joint] #gu.points_from_FR1_to_FR2(Mext_RS, original_3D_dict[joint])
            # new_3D_dict[joint] = gu.points_from_FR1_to_FR2(Mext_RS, original_3D_dict[joint])
        
        #Transform the extrinsic matrix for each camera from the original
        # Reference System to the reference_camera system
        Mext_new_dit = {}
        for cam in self.cam_list_rand:
            Mext_cam = self.camera_params[calib_idx][cam]['ext']
            Mext_new_dit[cam] = Mext_cam #gu.TR_from_camera_to_camera(Mext_RS, Mext_cam)
            # Mext_new_dit[cam] = gu.TR_from_camera_to_camera(Mext_cam, Mext_RS)

        #Create a cam_params dictionary with the new matrices
        new_cam_params = {}
        for cam in self.cam_list_rand:
            new_cam_params[cam] = {}
            new_cam_params[cam]['K'] = self.intr_matrices[calib_idx][cam]
            new_cam_params[cam]['distCoef'] = self.distCoefs[calib_idx][cam]
            new_cam_params[cam]['ext'] = Mext_new_dit[cam]
            new_cam_params[cam]['resolution'] = self.resolutions[calib_idx][cam]

        
        # precomputed_joints = self.frames[frame_idx]['precomputed_joints']
        precomputed_joints = du.generate_new_data(new_3D_dict, self.cam_list_rand, self.camera_params[calib_idx])
        joints_dict = {cam: precomputed_joints[cam] for cam in self.cam_list_rand if cam in precomputed_joints}


        #Create the dictionry with each canera information
        frame = {}
        N = len(self.cam_list_rand)
        L = len(self.joint_list)
        coords2D_virtual = torch.zeros((N, 3, L), dtype=torch.float32)
        for cam_idx, cam in enumerate(self.cam_list_rand):
            frame[cam] = [joints_dict[cam]]
            
            # Virtual cameras original 2D coords
            for joint_idx, joint in enumerate(self.joint_list):
                joint_str = str(joint)
                if cam in joints_dict and joint_str in joints_dict[cam]:
                    coords2D_virtual[cam_idx, 0, joint_idx] = float(joints_dict[cam][joint_str][1])
                    coords2D_virtual[cam_idx, 1, joint_idx] = float(joints_dict[cam][joint_str][2])
                    coords2D_virtual[cam_idx, 2, joint_idx] = float(joints_dict[cam][joint_str][3])


        frame = self.random_joints_removal(frame)
        apply_noise = self.training == True
        graph = HPE_Data(frame, new_3D_dict, new_cam_params, self.all_features, self.cam_list_rand, self.num_joints, self.norm_vector, self.use_3D_features, self.star_topology, apply_noise=apply_noise)

        original2D = {}

        # Original 2D coord
        coords2D_real = self.frames[frame_idx]['coords2D']
        if use_all:
            original2D["coord2D"] = torch.cat((coords2D_real, coords2D_virtual), dim=0)

            # Virtual camera params
            virtual_cams_extr_mat = torch.stack([torch.tensor(self.ext_matrices[calib_idx][cam],  dtype=torch.float32) for cam in self.cam_list_rand])
            virtual_cams_intr_mat = torch.stack([torch.tensor(self.intr_matrices[calib_idx][cam], dtype=torch.float32) for cam in self.cam_list_rand])
            virtual_cams_dist_coefs = torch.stack([torch.tensor(self.distCoefs[calib_idx][cam], dtype=torch.float32) for cam in self.cam_list_rand])
            
            original2D["extr"] = torch.cat((self.real_cams_extr_mat[calib_idx], virtual_cams_extr_mat), dim=0)
            original2D["intr"] = torch.cat((self.real_cams_intr_mat[calib_idx], virtual_cams_intr_mat), dim=0)
            original2D["distCoef"] = torch.cat((self.real_cams_dist_coefs[calib_idx], virtual_cams_dist_coefs), dim=0)
            

            # NCams
            NCams = len(self.real_cameras) + num_cameras
            original2D["NCams"] = torch.tensor(NCams, dtype=torch.int32)
        else:
            original2D["coord2D"] = coords2D_real

            original2D["extr"] = self.real_cams_extr_mat[calib_idx]
            original2D["intr"] = self.real_cams_intr_mat[calib_idx]
            original2D["distCoef"] = self.real_cams_dist_coefs[calib_idx]
            
            # NCams
            NCams = len(self.real_cameras)
            original2D["NCams"] = torch.tensor(NCams, dtype=torch.int32)

        return graph, original2D

    def get_item_supervised(self, frame_idx):
        num_cameras = random.randint(2, 5)
        self.cam_list_rand = random.sample(self.frames[frame_idx]['cams_with_2D'], k=num_cameras)   
        
        self.reference_camera = random.choice(self.cam_list_rand) 

        calib_idx = self.frames[frame_idx]['calib_idx']
        
        #Transform the coordenates 3D from the original Reference System
        # to the reference_camera system
        Mext_RS = self.camera_params[calib_idx][self.reference_camera]['ext']
        original_3D_dict = self.frames[frame_idx]['3D']
        new_3D_dict = {}
        for joint in original_3D_dict:
            new_3D_dict[joint] = original_3D_dict[joint] #gu.points_from_FR1_to_FR2(Mext_RS, original_3D_dict[joint])
            # new_3D_dict[joint] = gu.points_from_FR1_to_FR2(Mext_RS, original_3D_dict[joint])
        
        #Transform the extrinsic matrix for each camera from the original
        # Reference System to the reference_camera system
        Mext_new_dit = {}
        for cam in self.cam_list_rand:
            Mext_cam = self.camera_params[calib_idx][cam]['ext']
            Mext_new_dit[cam] = Mext_cam #gu.TR_from_camera_to_camera(Mext_RS, Mext_cam)
            # Mext_new_dit[cam] = gu.TR_from_camera_to_camera(Mext_cam, Mext_RS)

        #Create a cam_params dictionary with the new matrices
        new_cam_params = {}
        for cam in self.cam_list_rand:
            new_cam_params[cam] = {}
            new_cam_params[cam]['K'] = self.intr_matrices[calib_idx][cam]
            new_cam_params[cam]['distCoef'] = self.distCoefs[calib_idx][cam]
            new_cam_params[cam]['ext'] = Mext_new_dit[cam]
            new_cam_params[cam]['resolution'] = self.resolutions[calib_idx][cam]

        
        # precomputed_joints = self.frames[frame_idx]['precomputed_joints']
        precomputed_joints = du.generate_new_data(new_3D_dict, self.cam_list_rand, self.camera_params[calib_idx])
        joints_dict = {cam: precomputed_joints[cam] for cam in self.cam_list_rand if cam in precomputed_joints}


        #Create the dictionry with each canera information
        frame = {}
        N = len(self.cam_list_rand)
        L = len(self.joint_list)
        coords2D_virtual = torch.zeros((N, 3, L), dtype=torch.float32)
        for cam_idx, cam in enumerate(self.cam_list_rand):
            skeleton_list = []
            skeleton = joints_dict[cam]
            skeleton_list.append(skeleton) #Only one skeleton                
            frame[cam] = skeleton_list
            
        graph = HPE_Data(frame, new_3D_dict, new_cam_params, self.all_features, self.cam_list_rand, self.num_joints, self.norm_vector, self.use_3D_features, self.star_topology, apply_noise=True)

        GT = self.frames[frame_idx]['GT']


        return graph, GT
    
    def random_joints_removal(self, frame, rate=0.5, min_njoints=3):
        for cam in frame:
            skeleton_joints = list(frame[cam][0].keys())
            njoints = len(skeleton_joints)
            if njoints>min_njoints:
                njoints_removal = random.randint(0, int(njoints*0.5))
                if njoints_removal>0:
                    joints_to_remove = random.sample(skeleton_joints, njoints_removal)
                    for j in joints_to_remove:
                        del frame[cam][0][j]
        return frame
                
            


    def __len__(self):
        return len(self.frames)


def hpe_collate_fn(samples):
    """
    Custom collate function to handle batches of HPE_Data and metadata.

    This function overrides the default PyTorch collation to properly manage 
    the mixed output of HPE_Dataset.__getitem__, which returns both a 
    torch_geometric.data.Data object (graph) and a dictionary (metadata).

    Args:
        samples (list): A list of tuples as returned by the Dataset:
            [(graph_0, info_0), (graph_1, info_1), ..., (graph_n, info_n)]
            - graph (HPE_Data): The graph representation of the frame.
            - info (dict): Metadata including original coordinates and 
              camera parameters.

    Returns:
        tuple: A tuple containing:
            - batch_graph (torch_geometric.data.Batch): A large disjoint graph 
              containing all graphs in the batch. It concatenates 'x' features, 
              shifts 'edge_index' to avoid collisions, and creates the 'ptr' 
              vector for graph indexing.
            - infos (list): A list of the 'info' dictionaries for each sample 
              in the batch, preserved in their original format for evaluation 
              or error calculation.
    """
    
    graphs = []
    original2D = []
    for s in samples:
        graphs.append(s[0])
        original2D.append(s[1]["coord2D"])
    
    # graphs = [s[0] for s in samples]
    # infos = [s[1] for s in samples]
    batch_graph = Batch.from_data_list(graphs)
    batch_original2D = torch.cat(original2D, axis=2)
    return batch_graph, batch_original2D

def hpe_VR_collate_fn(samples):
    """
    Custom collate function to handle batches of HPE_Data and metadata.

    This function overrides the default PyTorch collation to properly manage 
    the mixed output of HPE_Dataset.__getitem__, which returns both a 
    torch_geometric.data.Data object (graph) and a dictionary (metadata).

    Args:
        samples (list): A list of tuples as returned by the Dataset:
            [(graph_0, info_0), (graph_1, info_1), ..., (graph_n, info_n)]
            - graph (HPE_Data): The graph representation of the frame.
            - info (dict): Metadata including original coordinates and 
              camera parameters.

    Returns:
        tuple: A tuple containing:
            - batch_graph (torch_geometric.data.Batch): A large disjoint graph 
              containing all graphs in the batch. It concatenates 'x' features, 
              shifts 'edge_index' to avoid collisions, and creates the 'ptr' 
              vector for graph indexing.
            - infos (list): A list of the 'info' dictionaries for each sample 
              in the batch, preserved in their original format for evaluation 
              or error calculation.
    """
    
    graphs = []
    original2D = []
    extr = []
    intr = []
    distCoef = []
    NCams = []
    for s in samples:
        graphs.append(s[0])
        original2D.append(s[1]["coord2D"])
        extr.append(s[1]["extr"])
        intr.append(s[1]["intr"])
        distCoef.append(s[1]["distCoef"])
        NCams.append(s[1]["NCams"])
        
    # graphs = [s[0] for s in samples]
    # infos = [s[1] for s in samples]
    batch_graph = Batch.from_data_list(graphs)
    batch_original2D = torch.cat(original2D, axis=0)
    batch_extr = torch.cat(extr, axis=0)
    batch_intr = torch.cat(intr, axis=0)
    batch_distCoef = torch.cat(distCoef, axis=0)
    batch_NCams = torch.tensor(NCams)
    return batch_graph, batch_original2D, batch_extr, batch_intr, batch_distCoef, batch_NCams

def hpe_supervised_collate_fn(samples):
    """
    Custom collate function for supervised learning.

    Args:
        samples (list): A list of tuples as returned by the Dataset:
            [(graph_0, GT_0), (graph_1, GT_1), ..., (graph_n, GT_n)]
            - graph (HPE_Data): The graph representation of the frame.
            - GT (dict): 3D position of all the joints.

    Returns:
        tuple: A tuple containing:
            - batch_graph (torch_geometric.data.Batch): A large disjoint graph 
              containing all graphs in the batch. It concatenates 'x' features, 
              shifts 'edge_index' to avoid collisions, and creates the 'ptr' 
              vector for graph indexing.
            - GT (torch.tensor): BxN tensor, with B the batch size and 
              N 3*number_of_joints.
    """
    
    graphs = []
    GT = []
    for s in samples:
        graphs.append(s[0])
        GT.append(s[1]["GT"])
        
    # graphs = [s[0] for s in samples]
    # infos = [s[1] for s in samples]
    batch_graph = Batch.from_data_list(graphs)
    batch_GT = torch.stack(GT)
    return batch_graph, batch_GT