import cv2
import numpy as np
from utils import geometric_utils as gu

def get_all_features(joint_list, use_3D_features):
    all_features = []

    # One-hot encoding
    # all_features.append('global_node')
    # all_features.append('cam_node')
    
    for joint in joint_list:
        joint_name = str(joint)
        
        all_features.append(joint_name + '_x2d')
        all_features.append(joint_name + '_y2d')
        all_features.append(joint_name + '_confidence')
        all_features.append(joint_name + '_visibility')
        
        if use_3D_features:
            # all_features.append(joint_name + '_line_pX')
            # all_features.append(joint_name + '_line_pY')
            # all_features.append(joint_name + '_line_pZ')
            # all_features.append(joint_name + '_line_vX')
            # all_features.append(joint_name + '_line_vY')
            # all_features.append(joint_name + '_line_vZ')
            all_features.append(joint_name + '_estimated')
            all_features.append(joint_name + '_x3d')
            all_features.append(joint_name + '_y3d')
            all_features.append(joint_name + '_z3d')
            
    for i in range(9):
        all_features.append('_intr_matrix_'+str(i))
    for i in range(12):
        all_features.append('_extr_matrix_'+str(i))
    for i in range(5):
        all_features.append('_distCoef_'+str(i))
    
    return all_features

def get_norm_vector(joint_list, use_3D_features):
    norm_dict = {
        'coord2D': {
            'W': 1920.0,
            'H': 1080.0 
        },
        
        'coord3D': 6500.0,
        
        'intr': [1862.69, 1.0, 1920.0, 1.0, 1862.69, 1080, 1.0, 1.0, 1.0], 
        
        'extr': [1.0, 1.0, 1.0, 6500, 1.0, 1.0, 1.0, 6500, 1.0, 1.0, 1.0, 6500], 
        
        'distCoef': [1.0, 1.0, 1.0, 1.0, 1.0]
    }
    
    norm_vector = []
    # norm_vector.append(1) #global_node
    # norm_vector.append(1) #cam_node
    for joint in joint_list:
        property_list = []
        
        property_list.append(norm_dict['coord2D']['W']) #x2d
        property_list.append(norm_dict['coord2D']['H']) #y2d
        property_list.append(1) #confidence
        property_list.append(1) #visiblity
        if use_3D_features:
            # property_list.append(1) #line_pX
            # property_list.append(1) #line_pY
            # property_list.append(1) #line_pZ
            # property_list.append(1) #line_vX
            # property_list.append(1) #line_vY
            # property_list.append(1) #line_vZ
            property_list.append(1) #estimated
            property_list.append(norm_dict['coord3D']) # x3d
            property_list.append(norm_dict['coord3D']) # y3d
            property_list.append(norm_dict['coord3D']) # z3d
    
        norm_vector.extend(property_list)
    
    norm_vector.extend(norm_dict['intr']) # intrinsic matrix
    norm_vector.extend(norm_dict['extr']) # extrinsic matrix
    norm_vector.extend(norm_dict['distCoef']) # distorsion coeficients
    
    return norm_vector

def generate_new_data(real3D, cam_list, cam_params):
    points3D = list(real3D.values())
    if not points3D:
        return None
    L = len(points3D)
    if L>1:
        points3D = np.concatenate(points3D, axis=1)
    else:
        points3D = points3D[0]
        
    Mext = list()
    Mint = list()
    distCoef = list()
    resolution = list()
    
    for cam in cam_list:
        Mext.append(cam_params[cam]['ext'])
        Mint.append(cam_params[cam]['K'])
        distCoef.append(cam_params[cam]['distCoef'])
        resolution.append(cam_params[cam]['resolution'])
    
    Mext = np.stack(Mext, axis=0)
    Mint = np.stack(Mint, axis=0)
    distCoef = np.stack(distCoef, axis=0)
    resolution = np.stack(resolution, axis=0).repeat(L, axis=-1).reshape((len(cam_list), 2, L))
    points2D = gu.project3D_numpy(Mext, Mint, distCoef, points3D)

    visible = np.logical_and(np.logical_and(points2D[:,0,:]>=0, points2D[:,1,:]>=0), np.logical_and(points2D[:,0,:]<resolution[:,0,:], points2D[:,1,:]<resolution[:,1,:]))
    visible = visible.astype(np.float64)

    # print(visible)
    
    #Create dictionary with the projected points
    pointsDict = dict()
    for i, cam in enumerate(cam_list):
        pointsDict[cam] = dict()
        for j, keypoint in enumerate(list(real3D.keys())):
            x, y, v = points2D[i, 0, j], points2D[i, 1, j], visible[i,j]
            idx = keypoint
            confidence = v
            pointsDict[cam][keypoint] = (idx, x, y, v, confidence)

    return pointsDict


def draw2DKeypoints(image, cam_keypoints, bones, cam_colors):
    for cam, skeleton in cam_keypoints.items():
        for idx in skeleton:
            coord = (int(skeleton[idx][1]), int(skeleton[idx][2]))
            # print(skeleton[idx])
            if skeleton[idx][3] > 0.5:
                color = cam_colors[cam]
            else:
                color = (180,180,180)
            
            try:
                cv2.circle(image, coord, 4, color, thickness=-1)
                cv2.putText(image, str(idx), coord, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)                  
            except Exception as e:
                pass

        # draw bones
        if skeleton.keys():
            tKey = type(list(skeleton.keys())[0])
            for idx in range(len(bones)):
                kp1 = tKey(bones[idx][0]-1)
                kp2 = tKey(bones[idx][1]-1)
                if kp1 in skeleton.keys() and kp2 in skeleton.keys():
                    coord1 = (int(skeleton[kp1][1]), int(skeleton[kp1][2]))
                    coord2 = (int(skeleton[kp2][1]), int(skeleton[kp2][2]))
                    if skeleton[kp1][3] > 0.5 and skeleton[kp2][3] > 0.5:
                        color = cam_colors[cam]
                    else:
                        color = (180,180,180)
                    
                    try:
                        if skeleton[kp1][3] > 0.5 and skeleton[kp2][3] > 0.5:
                            cv2.line(image, coord1, coord2, color, thickness=3)   
                    except Exception as e:
                        pass
