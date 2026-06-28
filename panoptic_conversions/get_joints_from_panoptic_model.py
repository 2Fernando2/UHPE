import sys
import os
import cv2
import numpy as np
import os.path
import copy
import json
import torchvision.transforms as transforms
import panutils
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
import yaml
import sys
import time
import pose_resnet
import torchvision.transforms as transforms
from string import ascii_lowercase
from typing import Dict, List, Tuple

def load_panoptic_model():
    config_file = "./cfg/prn64_cpn80x80x20_960x512_cam5.yaml"
    with open(config_file) as f:
        cfg = edict(yaml.load(f, Loader=yaml.FullLoader))


    backbone = pose_resnet.get_pose_net(cfg, is_train=True)

    model = backbone

    model = model.to('cuda')
    model.eval()
    return model

def get_output_from_panoptic_model(img, model):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tr = transforms.Compose([transforms.ToTensor(), normalize, ])

    image_size = (960, 512)
    img_input = cv2.resize(img, (960, 720))

    t = tr(img_input)

    with torch.no_grad():
        res = model(t[None, :].to('cuda'))
        res = res.to('cpu')

    return res

# import trt_pose.coco
# import trt_pose.plugins


def find_peaks(cmap, threshold, window_size, max_num_parts):
    # cmap shape: [1, C, H, W]
    device = cmap.device
    N, C, H, W = cmap.shape
    
    # NMS mediante Max Pooling
    pad = window_size // 2
    max_vals = F.max_pool2d(cmap, kernel_size=window_size, stride=1, padding=pad)
    peaks_mask = (cmap == max_vals) & (cmap > threshold)
    
    # Inicializar tensores de salida con ceros (como en C++)
    # peak_counts: [1, C]
    # peaks: [1, C, max_num_parts, 2]
    peak_counts = torch.zeros((N, C), dtype=torch.int32, device=device)
    peaks = torch.zeros((N, C, max_num_parts, 2), dtype=torch.float32, device=device)
    
    for c in range(C):
        # Encontrar �ndices de los picos para el canal c
        idx = torch.nonzero(peaks_mask[0, c]) # [num_found, 2] -> (y, x)
        num_found = idx.shape[0]
        
        # Limitar al m�ximo permitido
        actual_num = min(num_found, max_num_parts)
        peak_counts[0, c] = actual_num
        
        if actual_num > 0:
            # trt_pose guarda las coordenadas (y, x)
            peaks[0, c, :actual_num] = idx[:actual_num].float()
            
    return peak_counts, peaks

def refine_peaks(peak_counts, peaks, cmap, window_size):
    # peak_counts: [1, C]
    # peaks: [1, C, max_num_parts, 2]
    # cmap: [1, C, H, W]
    device = cmap.device
    N, C, max_num_parts, _ = peaks.shape
    H, W = cmap.shape[2:]
    pad = window_size // 2
    
    # Clonar para no modificar el original si se desea
    refined_peaks = torch.zeros_like(peaks)
    
    for c in range(C):
        num_peaks = peak_counts[0, c]
        for p in range(num_peaks):
            y_idx, x_idx = peaks[0, c, p].long()
            
            # Definir ventana local segura
            y_min, y_max = max(0, y_idx-pad), min(H, y_idx+pad+1)
            x_min, x_max = max(0, x_idx-pad), min(W, x_idx+pad+1)
            
            window = cmap[0, c, y_min:y_max, x_min:x_max]
            
            # Generar mallas de coordenadas
            yy, xx = torch.meshgrid(
                torch.arange(y_min, y_max, device=device), 
                torch.arange(x_min, x_max, device=device), 
                indexing='ij'
            )
            
            sum_w = window.sum()
            if sum_w > 0:
                # Centro de masa y Normalizaci�n (0.0 a 1.0)
                refined_y = (yy.float() * window).sum() / (sum_w * H)
                refined_x = (xx.float() * window).sum() / (sum_w * W)
                refined_peaks[0, c, p, 0] = refined_y
                refined_peaks[0, c, p, 1] = refined_x
            else:
                # Si falla, devolver posici�n original normalizada
                refined_peaks[0, c, p, 0] = y_idx.float() / H
                refined_peaks[0, c, p, 1] = x_idx.float() / W
                
    return refined_peaks


class ParseObjects(object):
    
    def __init__(self, topology, cmap_threshold=0.1, link_threshold=0.1, cmap_window=5, line_integral_samples=7, max_num_parts=100, max_num_objects=100):
        self.topology = topology
        self.cmap_threshold = cmap_threshold
        self.link_threshold = link_threshold
        self.cmap_window = cmap_window
        self.line_integral_samples = line_integral_samples
        self.max_num_parts = max_num_parts
        self.max_num_objects = max_num_objects
    
    def __call__(self, cmap):

        # peak_counts, peaks = trt_pose.plugins.find_peaks(cmap, self.cmap_threshold, self.cmap_window, self.max_num_parts)
        # normalized_peaks = trt_pose.plugins.refine_peaks(peak_counts, peaks, cmap, self.cmap_window)
        # cmap = torch.sigmoid(cmap)
        peak_counts, peaks = find_peaks(cmap, self.cmap_threshold, self.cmap_window, self.max_num_parts)
        normalized_peaks = refine_peaks(peak_counts, peaks, cmap, self.cmap_window)
        return normalized_peaks
    

def extract_heatmaps(model_output: torch.Tensor) -> torch.Tensor:
    if model_output.ndim == 4:
        if model_output.shape[0] != 1:
            raise RuntimeError(f"Batch no soportado: {tuple(model_output.shape)}")
        heatmaps = model_output[0]
    elif model_output.ndim == 3:
        heatmaps = model_output
    else:
        raise RuntimeError(f"Dimensión de salida no soportada: {tuple(model_output.shape)}")

    return heatmaps.float()


def refine_peak_subpixel(hm: torch.Tensor, y: int, x: int) -> Tuple[float, float]:
    h, w = hm.shape
    y0 = max(0, y - 1)
    y1 = min(h, y + 2)
    x0 = max(0, x - 1)
    x1 = min(w, x + 2)

    patch = torch.clamp(hm[y0:y1, x0:x1], min=0.0)
    s = float(patch.sum())
    if s <= 1e-12:
        return float(x), float(y)

    ys = torch.arange(y0, y1, dtype=patch.dtype, device=patch.device).view(-1, 1)
    xs = torch.arange(x0, x1, dtype=patch.dtype, device=patch.device).view(1, -1)

    cy = float((patch * ys).sum() / patch.sum())
    cx = float((patch * xs).sum() / patch.sum())
    return cx, cy


def parse_detected_joints(
    model_output: torch.Tensor,
    cam_resolution: Tuple[int, int],
    id_joint: Dict[int, str],
    peak_threshold: float = 0.15,
    nms_kernel: int = 5,
    max_peaks_per_joint: int = 100,
    apply_sigmoid: bool = True,
):
    heatmaps = extract_heatmaps(model_output)
    width, height = cam_resolution
    num_channels, hm_h, hm_w = heatmaps.shape

    # if apply_sigmoid:
    #     heatmaps = torch.sigmoid(heatmaps)

    pad = nms_kernel // 2
    pooled = F.max_pool2d(
        heatmaps.unsqueeze(0), kernel_size=nms_kernel, stride=1, padding=pad
    )[0]
    is_peak = (heatmaps == pooled) & (heatmaps >= peak_threshold)

    detected_joints = {}

    for j in range(min(num_channels, max(id_joint.keys()) + 1)):
        if j == 2:
            continue
        if j not in id_joint:
            continue

        joint_id = int(id_joint[j])
        ys, xs = torch.nonzero(is_peak[j], as_tuple=True)
        if ys.numel() == 0:
            continue

        scores = heatmaps[j, ys, xs]
        order = torch.argsort(scores, descending=True)[:max_peaks_per_joint]

        coords = []
        for idx in order.tolist():
            y = int(ys[idx])
            x = int(xs[idx])
            cx, cy = refine_peak_subpixel(heatmaps[j], y, x)

            x_img = (cx / max(1, hm_w - 1)) * (width - 1)
            y_img = (cy / max(1, hm_h - 1)) * (height - 1)
            coords.append([float(x_img), float(y_img)])

        if coords:
            detected_joints[joint_id] = coords

    return detected_joints


# -----------------------------
# Skeleton utilities
# -----------------------------

def coco19_to_coco18_map() -> Dict[int, str]:
    id_joint = {}
    id_joint[0] = '17'   # Neck
    id_joint[1] = '0'    # Nose
    # joint 2 is bodyCenter (not in coco18)
    id_joint[3] = '5'    # left shoulder
    id_joint[4] = '7'    # left elbow
    id_joint[5] = '9'    # left wrist
    id_joint[6] = '11'   # left hip
    id_joint[7] = '13'   # left knee
    id_joint[8] = '15'   # left ankle
    id_joint[9] = '6'    # right shoulder
    id_joint[10] = '8'   # right elbow
    id_joint[11] = '10'  # right wrist
    id_joint[12] = '12'  # right hip
    id_joint[13] = '14'  # right knee
    id_joint[14] = '16'  # right ankle
    id_joint[15] = '1'   # left eye
    id_joint[16] = '3'   # left ear
    id_joint[17] = '2'   # right eye
    id_joint[18] = '4'   # right ear
    return id_joint
    

with open('../utils/human_pose.json', 'r') as f:
    human_pose = json.load(f)
topology = None #trt_pose.coco.coco_category_to_topology(human_pose)
num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])
parse_objects = ParseObjects(topology, cmap_threshold=0.15, link_threshold=0.15)

draw = False

# Setup paths
seq_name = sys.argv[1]

if seq_name[-1] == '/':
    seq_name = seq_name[:-1]

hd_skel_json_path = seq_name+'/vgaPose3d_stage1_coco19/'

# Load camera calibration parameters
with open(seq_name+'/calibration_{0}.json'.format(seq_name.split('/')[-1])) as cfile:
    calib = json.load(cfile)

# Cameras are identified by a tuple of (panel#,node#)
cameras = {(cam['panel'],cam['node']):cam for cam in calib['cameras']}

# Convert data into numpy arrays for convenience
for k,cam in cameras.items():    
    cam['K'] = np.matrix(cam['K'])
    cam['distCoef'] = np.array(cam['distCoef'])
    cam['R'] = np.matrix(cam['R'])
    cam['t'] = np.array(cam['t']).reshape((3,1))


# Transform coco 19 into coco 18
id_joint = dict()
id_joint[0] = '17'  # Neck
id_joint[1] = '0'   # Nose
# joint 2 is bodyCenter (not in coco 18)
id_joint[3] = '5'   # left shoulder
id_joint[4] = '7'   # left elbow
id_joint[5] = '9'   # left wrist
id_joint[6] = '11'  # left hip
id_joint[7] = '13'  # left knee
id_joint[8] = '15'  # left ankle
id_joint[9] = '6'   # right shoulder
id_joint[10] = '8'  # right elbow
id_joint[11] = '10' # right wrist
id_joint[12] = '12' # right hip
id_joint[13] = '14' # right knee
id_joint[14] = '16' # right ankle
id_joint[15] = '1'  # left eye
id_joint[16] = '3'  # left ear
id_joint[17] = '2'  # right eye
id_joint[18] = '4'  # right ear 


# Access to the images' folder
cams_imgs_path = seq_name+'/vgaImgs/'

cam_directories = os.listdir(cams_imgs_path) #['00_03']
cam_directories.sort()


# Get images' paths and organize them
images_info = {}
for c in cam_directories:
    cam_id = c#int(c.split('_')[-1])
    imgs_path = os.path.join(cams_imgs_path, c)
    imgs = [f for f in os.listdir(imgs_path) if os.path.isfile(os.path.join(imgs_path, f))]
    imgs.sort()
    for img_name in imgs:
        img_id = img_name.split('.')[-2].split('_')[-1]
        if not img_id in images_info.keys():
            images_info[img_id] = {}
            images_info[img_id]['cameras'] = {}
            images_info[img_id]['json'] = os.path.join(hd_skel_json_path, 'body3DScene_'+img_id+'.json')
        images_info[img_id]['cameras'][cam_id] = os.path.join(imgs_path, img_name)

human_json = dict()
human_projected_json = dict()
cont = 0
model = load_panoptic_model()
sequence_len = len(images_info)
for id_frame, image in images_info.items():
    print(id_frame)
    if not os.path.exists(image['json']):
        continue

    print(cont,'/',sequence_len)

    # if cont>1000:
    #     break

    with open(image['json']) as dfile:
        bframe = json.load(dfile)

    # if not bframe['bodies']:
    #     continue
    
    cont += 1
    kps_per_human_and_cam = dict()
    kps_per_human_and_cam_projected = dict()

    for cam in image['cameras']:
        # Detection from image
        img_name = image['cameras'][cam]
        color_image = cv2.imread(img_name)
        ret = copy.deepcopy(color_image)
        
        panoptic_output = get_output_from_panoptic_model(color_image, model)

        panel_n, cam_n = int(cam.split('_')[0]), int(cam.split('_')[1])
        # Projection from 3D
        joints_3D = dict()
        projected_people = {}
        for body in bframe['bodies']:
            id_person = body['id']
            joints_3D[id_person] = dict()
            skel = np.array(body['joints19']).reshape((-1,4)).transpose()

            # Project skeleton into view (this is like cv2.projectPoints)
            pt = panutils.projectPoints(skel[0:3,:],
                        cameras[(panel_n, cam_n)]['K'], cameras[(panel_n,cam_n)]['R'], cameras[(panel_n,cam_n)]['t'], 
                        cameras[(panel_n, cam_n)]['distCoef'])


            # Show only points detected with confidence
            valid = skel[3,:]>0.1

            pt = pt.transpose()


            kps = dict()
            for i, joint in enumerate(pt):
                if not valid[i]:
                    continue
                if i!=2:
                    kp = id_joint[i]
                else:
                    kp = '-1'
                joints_3D[id_person][kp] = [float(skel[0][i]), float(skel[1][i]), float(skel[2][i])]
                if joint[0] < 0 or joint[0] >= cameras[(panel_n,cam_n)]['resolution'][0] or joint[1] < 0 or joint[1] >= cameras[(panel_n,cam_n)]['resolution'][1]:
                    continue
                x = joint[0]
                y = joint[1]
                kps[int(kp)] = [int(kp), float(x), float(y), 1, 1]
                if draw:
                    cv2.circle(ret, (int(x), int(y)), 1, (0, 0, 255), 2)
                    cv2.putText(ret, "%d" % int(kp), (int(x) + 5, int(y)),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            projected_people[id_person] = copy.deepcopy(kps)

        # Organize detected joints into separate skeletons according to the their proximity to projected people

        detected_joints = dict()

        detected_joints = parse_detected_joints(
            panoptic_output,
            tuple(cameras[panel_n, cam_n]['resolution']),
            id_joint,
            peak_threshold=0.15,
            nms_kernel=3,
            max_peaks_per_joint=50,
            apply_sigmoid=False,
        )

        # if draw:
        #     for j, coords in detected_joints.items():
        #         for coord in coords:
        #             x, y = coord[0], coord[1]
        #             cv2.circle(ret, (int(x), int(y)), 1, (0, 255, 0), 2)
        #             if j!=2:
        #                 cv2.putText(ret, "%d" % int(j), (int(x) + 5, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)



        # peaks = parse_objects(panoptic_output)


        # for j, person in enumerate(peaks[0]):
        #     if torch.count_nonzero(person) == 0:
        #         break
        #     if j == 2:
        #         continue
        #     idx = int(id_joint[j])
        #     detected_joints[idx] = list()
        #     for kp in person:
        #         if torch.count_nonzero(kp) == 0:
        #             continue
        #         y = kp[0] * cameras[(panel_n, cam_n)]['resolution'][1]
        #         x = kp[1] * cameras[(panel_n, cam_n)]['resolution'][0]
        #         detected_joints[idx].append([x, y])
        #         if draw:
        #             cv2.circle(ret, (int(x), int(y)), 1, (0, 255, 0), 2)
        #             if j!=2:
        #                 cv2.putText(ret, "%d" % int(idx), (int(x) + 5, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        # print(detected_joints)                        

        detected_people = dict()
        projected_filtered_people = dict()
        for id_person, skeleton in projected_people.items():
            kps = dict()
            kps_projected = dict()
            for j, joint in skeleton.items(): 
                if j in detected_joints.keys():
                    p2D = np.array(joint[1:3])
                    min_dist = 100000000
                    nearest = None
                    for i, coor in enumerate(detected_joints[j]):
                        d2D = np.array(coor)
                        dist = np.linalg.norm(p2D - d2D)
                        if dist < min_dist:
                            min_dist = dist
                            nearest = coor
                    if min_dist < 15:
                        kps[j] = [j, float(nearest[0]), float(nearest[1]), 1, 1]
                        kps_projected[j] = [j, float(p2D[0]), float(p2D[1]), 1, 1]
            if kps:
                detected_people[id_person] = copy.deepcopy(kps)
                projected_filtered_people[id_person] = copy.deepcopy(kps_projected)


        for id_person in detected_people:
            data = detected_people[id_person]
            data_prj = projected_filtered_people[id_person]
            cam_name = 'vga_'+cam
            if not id_person in kps_per_human_and_cam.keys():
                kps_per_human_and_cam[id_person] = dict()
            if not id_person in kps_per_human_and_cam_projected.keys():
                kps_per_human_and_cam_projected[id_person] = dict()

            kps_per_human_and_cam[id_person][cam_name] = [json.dumps([data]), time.time(), 'no_image', [joints_3D[id_person]]]    
            kps_per_human_and_cam_projected[id_person][cam_name] = [json.dumps([data_prj]), time.time(), 'no_image', [joints_3D[id_person]]]    
            if draw:
                for idx, joint in data.items():
                    x = joint[1]
                    y = joint[2]
                    cv2.circle(ret, (int(x), int(y)), 1, (0, 255, 0), 2)
                    cv2.putText(ret, "%d" % int(idx), (int(x) + 5, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)


        if draw:
            cv2.imshow("test", cv2.resize(ret, dsize=None, fx=1, fy=1))
            if cv2.waitKey(0) % 256 == 27:
                print("Escape hit, closing...")
                cv2.destroyAllWindows()
                sys.exit(0)

    for id_person in kps_per_human_and_cam.keys():
        if not id_person in human_json.keys():
            human_json[id_person] = list()

        human_json[id_person].append(dict(kps_per_human_and_cam[id_person]))

    for id_person in kps_per_human_and_cam_projected.keys():
        if not id_person in human_projected_json.keys():
            human_projected_json[id_person] = list()

        human_projected_json[id_person].append(dict(kps_per_human_and_cam_projected[id_person]))


final_json = []
for p, j in human_json.items():
    final_json += j

output_file = open(seq_name + '_detected.json', 'w')
output_file.write(json.dumps(final_json))
output_file.close()

final_projected_json = []
for p, j in human_projected_json.items():
    final_projected_json += j

output_file = open(seq_name + '_projected.json', 'w')
output_file.write(json.dumps(final_projected_json))
output_file.close()
