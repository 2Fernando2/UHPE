import os
import shutil
import random
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GATNetwork for Human Pose Estimation')

    parser.add_argument('--paramsdir', type=str, required=True, help='Directory containing camera calibration files')
    parser.add_argument('--traindir', type=str, required=True, help='Directory where training files will be saved')
    parser.add_argument('--testdir', type=str, required=True, help='Directory where test files will be saved')
    parser.add_argument('--num_excluded_cameras', type=int, required=True, help='Number of cameras to discard for each panel')

    args = parser.parse_args()
    num_excluded_cameras = args.num_excluded_cameras
    
    # Check if traindir is empty
    if os.path.isdir(args.traindir) and len(os.listdir(args.traindir)) > 0:
        answer = input("traindir already exists. Do you want to remove its contents (Y/n)?")
        if answer.lower() in ["", "n"]:
            print("exit")
            exit(0)
        else:
            print("Removing traindir contents...")
            # Remove the directory and its contents
            shutil.rmtree(args.traindir)
            # Recreate the directorty
            os.makedirs(args.traindir)
    
    # Check if testdir is empty
    if os.path.isdir(args.testdir) and len(os.listdir(args.testdir)) > 0:
        answer = input("testdir already exists. Do you want to remove its contents (Y/n)?")
        if answer.lower() in ["", "n"]:
            print("exit")
            exit(0)
        else:
            print("Removing testdir contents...")
            # Remove the directory and its contents
            shutil.rmtree(args.testdir)
            # Recreate the directorty
            os.makedirs(args.testdir)
    
    if num_excluded_cameras not in [1, 2]:
        raise ValueError("num_excluded_cameras must be 1 or 2")
    
    cam_idx_list = os.listdir(args.paramsdir)
    original_id_cams = [f.replace('calib_', '').replace('.json', '') for f in cam_idx_list if f.startswith('calib_')]
    
    # Discard num_excluded_cameras for each panel
    # Cameras from the internal ring
        #   Cameras 19, 20, 21, 22, 23, 24
    internal_cameras = ["19", "20", "21", "22", "23", "24"]
    # Cameras from the external ring
        #   Cameras 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18
    external_cameras = ["02", "03", "05", "06", "08", "09", "11", "12", "14", "15", "17", "18"]
    
    excluded_cameras = []
    for panel_id in range(1, 21):
        # Choose one camera from the internal ring
        internal_camera_id = random.choice(internal_cameras)
        internal_camera_name = f"vga_{panel_id:02d}_{internal_camera_id}"
    
        # Choose one camera from the external ring
        external_camera_id = random.choice(external_cameras)
        external_camera_name = f"vga_{panel_id:02d}_{external_camera_id}"

        # Exclude only num_excluded_cameras
        if num_excluded_cameras == 1:
            excluded_camera = random.choice([internal_camera_name, external_camera_name])
            excluded_cameras.append(excluded_camera)
        else:
            excluded_cameras.append(internal_camera_name)
            excluded_cameras.append(external_camera_name)

    excluded_set = set(excluded_cameras)
    # Remove excluded cameras from original list
    train_cams = [cam for cam in original_id_cams if cam not in excluded_set]
    # Extract the excluded cameras 
    test_cams = [cam for cam in original_id_cams if cam in excluded_set]
    
    # Create directories if they don't exist
    os.makedirs(args.traindir, exist_ok=True)
    os.makedirs(args.testdir, exist_ok=True)
    
        # Copy train files
    print(f"Copying {len(train_cams)} camera files to train directory...")
    for cam in train_cams:
        src_file = os.path.join(args.paramsdir, f'calib_{cam}.json')
        dst_file = os.path.join(args.traindir, f'calib_{cam}.json')
        shutil.copy2(src_file, dst_file)
    
    # Copy test files
    print(f"Copying {len(test_cams)} camera files to test directory...")
    for cam in test_cams:
        src_file = os.path.join(args.paramsdir, f'calib_{cam}.json')
        dst_file = os.path.join(args.testdir, f'calib_{cam}.json')
        shutil.copy2(src_file, dst_file)
    
    print(f"\nSplit completed successfully!")
    print(f"Train cameras: {len(train_cams)}")
    print(f"Test cameras: {len(test_cams)}")