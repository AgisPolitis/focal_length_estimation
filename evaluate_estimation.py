import os
import argparse
import numpy as np
from estimate_focal_lengths import merge_focal_lengths, get_file_paths

def get_immediate_subfolders(folder_path):
    subfolders = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            subfolders.append(item_path)
    return subfolders

def read_intrinsics_bundlefusion(gt_intrinsics_path):
    """
    Extracts the intrinsics parameters from the specific
    file according to the bundlefusion format
    """
    with open(gt_intrinsics_path, 'r') as file:
        for line in file:
            if line.startswith('m_calibrationColorIntrinsic'):
                intrinsics = line.split()
                fx, fy, cx, cy = intrinsics[2], intrinsics[7], intrinsics[4], intrinsics[8]
        print(float(fx), float(fy), float(cx), float(cy))
    return float(fx), float(fy), float(cx), float(cy)

def read_intrinsics_arkit(gt_intrinsics_path):
    """
    Extracts the intrinsics parameters from the specific
    file according to the arkitscenes format
    """
    # Read the file and extract the last four numbers
    with open(gt_intrinsics_path, 'r') as file:
        line = file.readline().strip()
        intrinsics = line.split()
        fx, fy, cx, cy = intrinsics[2], intrinsics[3], intrinsics[4], intrinsics[5]
    return float(fx), float(fy), float(cx), float(cy)

def evaluate_bundlefusion_scene(images_directory, gt_intrinsics_path):
    """
    Runs the K estimation procedure in a given image directory
    and computes the absolute error compared to the ground truth intrinsics
    """
    fx, fy, cx, cy = merge_focal_lengths(images_directory)
    fx_gt, fy_gt, cx_gt, cy_gt = read_intrinsics_bundlefusion(gt_intrinsics_path)

    print("------------------------------------------------")
    print(f"Fx error on the scene is: {np.abs(fx-fx_gt)/fx_gt}")
    print(f"Fy error on the scene is: {np.abs(fy-fy_gt)/fy_gt}")
    print("------------------------------------------------")
    return np.abs(fx-fx_gt)/fx_gt, np.abs(fy-fy_gt)/fy_gt

def evaluate_bundlefusion_whole(dataset_directory):
    """
    Evaluates the K estimation procedure in many
    scenes of the bundlefusion dataset
    """
    # Get the list of each scene's path
    scenes_paths = get_immediate_subfolders(dataset_directory)

    # Evaluate for every scene
    errors = []
    for scene_path in scenes_paths:
        scene_id = scene_path.split("/")[-1]
        images_directory = scene_path + "/images/"
        gt_intrinsics_path = scene_path + "/" + "info.txt"

        # Get errors for this scene
        print(" ---------------- PROCESSING SCENE: " + scene_id + " ----------------")
        error_fx, error_fy = evaluate_bundlefusion_scene(images_directory, gt_intrinsics_path)
        errors.append([error_fx, error_fy])
    
    errors = np.array(errors)

    # Get mean errors (its percetnage error compared to gt focal lengths)
    mae_fx = np.mean(errors[:, 0])
    mae_fy = np.mean(errors[:, 1])

    print("------------------------------------------------")
    print(f"Total MAE for fx is: {mae_fx}")
    print(f"Total MAE for fy is: {mae_fy}")
    print("------------------------------------------------")
    return

def evaluate_arkitscences_scene(images_directory, gt_intrinsics_path):
    """
    Runs the K estimation procedure in a given image directory
    and computes the absolute error compared to the ground truth intrinsics
    """
    fx, fy, cx, cy = merge_focal_lengths(images_directory)
    fx_gt, fy_gt, cx_gt, cy_gt = read_intrinsics_arkit(gt_intrinsics_path)

    print("------------------------------------------------")
    print(f"Fx error on the scene is: {np.abs(fx-fx_gt)/fx_gt}")
    print(f"Fy error on the scene is: {np.abs(fy-fy_gt)/fy_gt}")
    print("------------------------------------------------")
    return np.abs(fx-fx_gt)/fx_gt, np.abs(fy-fy_gt)/fy_gt

def evaluate_arkitscenes_whole(dataset_directory):
    """
    Evaluates the K estimation procedure in many
    scenes of the arkitscences dataset
    """
    # Get the list of each scene's path
    scenes_paths = get_immediate_subfolders(dataset_directory)
    
    # Evaluate for every scene
    errors = []
    for scene_path in scenes_paths:
        scene_id = scene_path.split("/")[-1]
        images_directory = scene_path + "/" + scene_id + "_frames/lowres_wide/"
        
        # Just get the first file for intrinsics (all are the same)
        gt_intrinsics_path = get_file_paths(scene_path + "/" + scene_id + "_frames/lowres_wide_intrinsics/")[0]

        # Get errors for this scene
        print(" ---------------- PROCESSING SCENE: " + scene_id + " ----------------")
        error_fx, error_fy = evaluate_arkitscences_scene(images_directory, gt_intrinsics_path)
        errors.append([error_fx, error_fy])

    errors = np.array(errors)

    # Get mean errors
    mae_fx = np.mean(errors[:, 0])
    mae_fy = np.mean(errors[:, 1])

    print("------------------------------------------------")
    print(f"Total MAE for fx is: {mae_fx}")
    print(f"Total MAE for fy is: {mae_fy}")
    print("------------------------------------------------")
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='Directory of the images of the scene')
    parser.add_argument('--eval_dataset', type=str, help='Which dataset to evaluate')
    
    args = parser.parse_args()

    if args.eval_dataset == "arkitscenes":
        evaluate_arkitscenes_whole(args.dataset_dir)
    elif args.eval_dataset == "bundlefusion":
        evaluate_bundlefusion_whole(args.dataset_dir)
    else:
        print("INVALID OPTION, available options for eval_dataset are 1. arkitscenes, 2. bundlefusion")
    