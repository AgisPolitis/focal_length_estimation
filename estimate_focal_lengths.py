import os
import argparse
import numpy as np
import cv2 as cv
from tqdm import tqdm
import matplotlib.pyplot as plt

from geometry import match_image_pair, find_fundamental_matrix, decompose_focal_length


def get_file_paths(directory):
    """
    Returns a list of all the image paths (ordered)
    in a given directory 
    """
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            file_paths.append(os.path.join(root, filename))
    file_paths.sort()
    return file_paths

def image_pair_focal_lengths(image0_path, image1_path):
    """
    Given a pair of images calculates the focal leghts 
    of the camera and the principal points
    """

    # Find point correspondences between images (also get principal point coords)
    try:
        points0, points1, cx, cy = match_image_pair(image0_path, image1_path)
    except IndexError: 
        return "index_error", -1, -1, -1

    # Calculate fundamental matrix   
    F = find_fundamental_matrix(points0, points1, method=cv.USAC_ACCURATE)
   
    # Possible error in some cv2 operations
    if type(F) == str:
        return "CV2_error", -1, -1, -1
    if F is None:
        return "None_F", -1, -1, -1
    # may return F matrix of shape (9, 3)
    if F.shape != (3,3):
        return "wrong_shape", -1, -1, -1

    # Decompose F to get focal lengths
    fx, fy = decompose_focal_length(F, cx, cy)
    return fx, fy, cx, cy

def merge_focal_lengths(directory, method="median"):
    """
    Calculates the focal length of the scene's camera
    by merging the focal lengths computed by sequential image
    pairs of the scene.
    """
    paths = get_file_paths(directory)

    # Calculate focal lenghts pairs of images
    estimated_focal_lengths = []

    stride = 5 # can be adjusted for skipping more or less frames
    num_throw_images = 2 # can be adjusted for using less images to reduce execution time (may harm performance!)

    print("[INFO] Starting focal length estimation ...")
    print(f"[INFO] Using stride = {stride} and num_throw_images = {num_throw_images}")
    
    for i in tqdm(range(round(len(paths)/num_throw_images) - stride+1)): # compare non-consecutive frames
        image0_path = paths[i]
        image1_path = paths[i + stride]

        fx, fy, cx, cy = image_pair_focal_lengths(image0_path, image1_path)

        # Check for possible exception 
        if fx != "singular" and fx != "wrong_shape" and fx != "None_F" and fx != "index_error" and fx != "CV2_error":
            estimated_focal_lengths.append([fx, fy]) 

    estimated_focal_lengths = np.array(estimated_focal_lengths)

    # Merge all the computed focal lengths
    if method == "median":
        fx = np.nanmedian(estimated_focal_lengths[:, 0])
        fy = np.nanmedian(estimated_focal_lengths[:, 1])
    else:  # mean
        fx = np.nanmean(estimated_focal_lengths[:, 0])
        fy = np.nanmean(estimated_focal_lengths[:, 1])
    
    return fx, fy, cx, cy
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='Directory of the images of the scene')
    args = parser.parse_args()

    fx, fy, cx, cy = merge_focal_lengths(args.input_dir, method="median")
    print(fx, fy, cx, cy)
   

    
            

