import torch
import matplotlib.pyplot as plt
import cv2 as cv 
import numpy as np
from numpy.linalg import LinAlgError

from LightGlue.lightglue import LightGlue, SuperPoint
from LightGlue.lightglue.utils import load_image, rbd
from LightGlue.lightglue import viz2d


def match_image_pair(image0_path, image1_path):
    """
    Matches an image pair using SuperPoint + LightGlue
    (https://github.com/cvg/LightGlue)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # SuperPoint+LightGlue
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features='superpoint', filter_threshold=0.1).eval().to(device)

    # Load each image as a torch.Tensor on GPU (optional) with shape (3,H,W), normalized in [0,1]
    image0 = load_image(image0_path).to(device)
    image1 = load_image(image1_path).to(device)
    
    # Calculate image dimensions
    h, w = image0.shape[1:]

    # Extract local features
    feats0 = extractor.extract(image0, resize=None)
    feats1 = extractor.extract(image1, resize=None)

    # Match the features
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
    matches = matches01['matches']  # indices with shape (K,2)

    # Get only the matched keypoints
    points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
    points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)

    return points0.cpu(), points1.cpu(), w/2, h/2

def find_fundamental_matrix(points0, points1, method=cv.USAC_ACCURATE):
    """
    Calculates the fundamental matrix between two views
    using the GraphCutRansac method described in (https://github.com/danini/graph-cut-ransac)
    """
    points0 = np.array(points0)
    points1 = np.array(points1)
    try:
        F, mask = cv.findFundamentalMat(points0, points1, method, 1) # last parameter is reproj threshold
    except cv.error:
        return "CV2_error"
    return F

def decompose_focal_length(F, cx, cy):
    """
    Given the fundamental matrix between 2 views
    and the principal point coordinates calculates
    the focal lengths of the camera. We assume that the 
    cameras have the same principal point coordinates.
    """
    K1 = np.array([[1, 0, cx],
                   [0, 1, cy],
                   [0, 0, 1]])

    K2 = np.array([[1, 0, cx],
                   [0, 1, cy],
                   [0, 0, 1]])
    
    F_normalized = np.dot(np.transpose(K2), np.dot(F, K1))
    
    U, _, V = np.linalg.svd(F_normalized, full_matrices=False)

    # To match the C++ JacobiSVD implementation 
    U = -1 * U
    V = -1 * np.transpose(V)

    e1 = V[:, -1]
    e2 = U[:, -1]

    sc1 = np.sqrt(e1[0] * e1[0] + e1[1] * e1[1])
    R1 = np.array([[e1[0] / sc1, e1[1] / sc1, 0],
                   [-e1[1] / sc1, e1[0] / sc1, 0],
                   [0, 0, 1]])
    Re1 = np.dot(R1, e1)

    sc2 = np.sqrt(e2[0] * e2[0] + e2[1] * e2[1])
    R2 = np.array([[e2[0] / sc2, e2[1] / sc2, 0],
                   [-e2[1] / sc2, e2[0] / sc2, 0],
                   [0, 0, 1]])
    Re2 = np.dot(R2, e2)

    RF = np.dot(R2, np.dot(F_normalized, np.transpose(R1)))
  
    C2 = np.zeros((3,3))
    C2[0,0], C2[1,1], C2[2,2] = Re2[2], 1, -Re2[0]
    C1 = np.zeros((3,3))
    C1[0,0], C1[1,1], C1[2,2] = Re1[2], 1, -Re1[0]

    # Catch a singular matric case 
    try:
        A = np.dot(np.linalg.inv(C2), np.dot(RF, np.linalg.inv(C1)))
    except LinAlgError as e:
        if "Singular matrix" in str(e):
            return "singular", "singular"
        else:
            return "error", "error"

    a, b, c, d = A[0,0], A[0,1], A[1,0], A[1,1]

    ff1 = -a * c * Re1[0] * Re1[0] / (a * c * Re1[2] * Re1[2] + b * d)
    f1 = np.sqrt(ff1)

    ff2 = -a * b * Re2[0] * Re2[0] / (a * b * Re2[2] * Re2[2] + c * d)
    f2 = np.sqrt(ff2)

    return f1, f2
