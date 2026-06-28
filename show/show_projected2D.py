import os
import sys
import json
import numpy as np
import argparse
import cv2
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
import utils.geometric_utils as gu

class projected2DTester():
    def __init__(self, json_file, paramsdir, joint_list, axes_3D):
        self.input_data = json.load(open(json_file, 'rb'))

        self.paramsdir = paramsdir
        self.joint_list = joint_list
        self.axes_3D = axes_3D

        self.get_camera_params()

    def get_camera_params(self):
        #Get camera names

        self.camera_names = []
        camera_files = os.listdir(self.paramsdir)
        for f in camera_files:
            if f.startswith('calib_') and f.endswith('.json'):
                cam = f.split('.json')[0].split('calib_')[-1]
                self.camera_names.append(cam)
    
        self.camera_params = {}
        self.camera_matrices = {}
        self.distortion_coefficients = {}
        self.projection_matrices = {}
        self.ext_matrices = {}
        self.resolution = {}
        for cam in self.camera_names:
            params_file = os.path.join(self.paramsdir, 'calib_'+cam+'.json')
            cam_params = json.load(open(params_file, 'rb'))
            self.camera_params[cam] = gu.get_parameters(cam_params)
            self.camera_matrices[cam] = self.camera_params[cam]['K']
            self.distortion_coefficients[cam] = self.camera_params[cam]['distCoef']
            self.projection_matrices[cam] = self.camera_params[cam]['proj']
            self.ext_matrices[cam] = self.camera_params[cam]['ext']
            self.resolution[cam] = self.camera_params[cam]['resolution']
            
    def estimate3D(self, itert):
        if itert>=len(self.input_data):
            return None
        
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
        return result3D
    
    def project3D(self, result3D, cameras):
        points3D = list(result3D.values())
        if not points3D:
            return None
        L = len(points3D)
        if L>1:
            points3D = np.concatenate(points3D, axis=1)
        else:
            points3D = points3D[0]

        Mext = list()
        Mint = list()
        kd = list()
        res = list()
        for cam in cameras:
            Mint.append(self.camera_matrices[cam])
            Mext.append(self.ext_matrices[cam])
            kd.append(self.distortion_coefficients[cam])
            res.append(self.resolution[cam])
        Mint = np.stack(Mint, axis=0)
        Mext = np.stack(Mext, axis=0)
        kd = np.stack(kd, axis=0)
        res = np.stack(res, axis=0).repeat(L, axis=-1).reshape((len(cameras),2,L))
        points2D = gu.project3D_numpy(Mext, Mint, kd, points3D)

        visible = np.logical_and(np.logical_and(points2D[:,0,:]>=0, points2D[:,1,:]>=0), np.logical_and(points2D[:,0,:]<res[:,0,:], points2D[:,1,:]<res[:,1,:]))

        print(visible)

        # create dictionary with the projected points
        pointsDict = dict()
        for i, cam in enumerate(cameras):
            pointsDict[cam] = dict()
            for j, kp in enumerate(list(result3D.keys())):
                x, y, v = points2D[i, 0, j], points2D[i, 1, j], visible[i,j]
                pointsDict[cam][kp] = (x,y,v)

        return pointsDict


def draw2DKeypoints(image, cam_keypoints, bones, cam_colors):
    for cam, skeleton in cam_keypoints.items():
        for idx in skeleton:
            coord = (int(skeleton[idx][0]), int(skeleton[idx][1]))
            if skeleton[idx][2]:
                color = cam_colors[cam]
            else:
                color = (180,180,180)
            cv2.circle(image, coord, 4, color, thickness=-1)   
            cv2.putText(image, str(idx), coord, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)                  

        # draw bones
        for idx in range(len(bones)):
            kp1 = str(bones[idx][0]-1)
            kp2 = str(bones[idx][1]-1)
            if kp1 in skeleton.keys() and kp2 in skeleton.keys():
                coord1 = (int(skeleton[kp1][0]), int(skeleton[kp1][1]))
                coord2 = (int(skeleton[kp2][0]), int(skeleton[kp2][1]))
                if skeleton[kp1][2] and skeleton[kp2][2]:
                    color = cam_colors[cam]
                else:
                    color = (180,180,180)
                cv2.line(image, coord1, coord2, color, thickness=3)   



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Display 3D multi-pose results using triangulation')

    parser.add_argument('--testfile', type=str, required=True, help='Test file used as input')
    parser.add_argument('--paramsdir', type=str, nargs='?', required=True, help='Directory that contains the camera parameters files')
    parser.add_argument('--width', type=int, default=1920, help='Image width')
    parser.add_argument('--height', type=int, default=1080, help='Image height')


    args = parser.parse_args()

    TEST_FILE = args.testfile
    PARAMSDIR = args.paramsdir
    width = args.width
    height = args.height

    joint_list = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    camera_colours={'hd_00_03': (255, 0, 0), 'hd_00_06': (0, 255, 0), 'hd_00_12': (0, 0, 255),
                    'hd_00_13': (127, 127, 0), 'hd_00_23': (0, 127, 127)}
    axes_3D = {'X': (0, 1.), 'Y': (2, 1.), 'Z': (1, -1.)}


    prj_tester = projected2DTester(TEST_FILE, PARAMSDIR, joint_list, axes_3D)
    new_cams = random.sample(prj_tester.camera_names, k=5)
    new_camera_colours={}
    for i, c in enumerate(new_cams):
        color = camera_colours[list(camera_colours.keys())[i]]
        new_camera_colours[c] = color

    path = str(ROOT) + "/utils/human_pose.json"
    with open(path, 'r') as f:
        human_pose = json.load(f)
        bones = human_pose["skeleton"]

    i = -1

    while True:
        i+=1
        prj_image = np.full((height, width, 3), 255, dtype=np.uint8)
        estimated3D = prj_tester.estimate3D(i)
        if  estimated3D is not None:
            projected2D = prj_tester.project3D(estimated3D, new_cams)
            if projected2D is not None:
                draw2DKeypoints(prj_image, projected2D, bones, new_camera_colours)
                cv2.imshow("Projected 2D", prj_image)
                k = cv2.waitKey(0)
                if k%256 == 27:
                    print("Escape hit, closing...")
                    cv2.destroyAllWindows()
                    break

