import sys
import numpy as np
import cv2
import json
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create camera parameters\' files from a Panoptic calibration file')
    parser.add_argument('--sequencefile', type=str, required=True, help='Input file')
    parser.add_argument('--width', type=int, default=1920, help='Image width')
    parser.add_argument('--height', type=int, default=1080, help='Image height')

    args = parser.parse_args()

    width = args.width
    height = args.height

    picture = np.full((height,width,3), 255, dtype=np.uint8)

    cv2.namedWindow("projection", cv2.WINDOW_AUTOSIZE)
    # cv2.namedWindow("projection", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("projection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("projection", picture)

    data = json.load(open(args.sequencefile, 'r'))

    joint_list = [x for x in range(18)]

    camera_colours={'hd_00_03': (255, 0, 0), 'hd_00_06': (0, 255, 0), 'hd_00_12': (0, 0, 255),
                    'hd_00_13': (127, 127, 0), 'hd_00_23': (0, 127, 127)}

    with open("./human_pose.json", 'r') as f:
        human_pose = json.load(f)
        bones = human_pose["skeleton"]
        keypoints = human_pose["keypoints"]


    for datum in data:
        # print(datum)
        number_of_cameras_with_data = 0
        picture = np.full((height, width, 3), 255, dtype=np.uint8)

        human = {}
        for k in datum.keys(): #parameters.camera_names:
            if len(datum[k]) == 3:
                d, _timestamp, other = datum[k]
            elif len(datum[k]) == 4:
                d, _timestamp, other, joints_3d = datum[k]
            else:
                print('Error in data format')
                exit()
            received_d = json.loads(d)
            # If there are multiple skeletons, we get the one with more data, as it's more likely to be real
            human = {}
            if len(received_d)>1:
                print(k, len(received_d))
            people_joints = []
            for received in received_d:
                n_joints = 0
                skeleton = {}
                # draw keypoints
                for idx_i in joint_list:
                    idx = str(idx_i)
                    try:
                        joint_data = received[idx]
                        if joint_data[3]>0.5:
                            n_joints+=1
                        coord = tuple([int(round(xx)) for xx in joint_data[1:3]])
                        skeleton[idx_i] = coord
                        cv2.circle(picture, coord, 4, camera_colours[k], thickness=-1)   
                        cv2.putText(picture, str(idx), coord, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)                  
                        # if idx_i in [0, 1, 3, 5, 7, 9, 11, 13, 15, 17]:
                        #     cv2.circle(picture, coord, 4, parameters.camera_colours[k], thickness=-1)
                        # else:
                        #     cv2.circle(picture, coord, 4, (0, 0, 0), thickness=-1)
                    except KeyError:
                        pass
                # draw bones
                for idx in range(len(bones)):
                    kp1 = bones[idx][0]-1
                    kp2 = bones[idx][1]-1
                    if kp1 in skeleton.keys() and kp2 in skeleton.keys():
                        cv2.line(picture, skeleton[kp1], skeleton[kp2], camera_colours[k], thickness=3)   
                people_joints.append(n_joints)
                if len(received) > len(human):
                    human = received
            print('cam', k)
            print(people_joints)

            if len(human)>0:
                number_of_cameras_with_data += 1


        cv2.putText(picture, str(number_of_cameras_with_data), (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1., (0,0,0), 1, cv2.LINE_AA) 

        cv2.imshow("projection", picture)
        k = cv2.waitKey(0)
        if k%256 == 27:
            print("Escape hit, closing...")
            cv2.destroyAllWindows()
            sys.exit(0)





