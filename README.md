# Focal length estimation from sequential images captured from an uknown camera

This repository contains a tool for estimating the focal length of an uknown camera, using a collection of sequential images captured with that camera. It can be used for estimating the intrinsics of a camera that was used to capture an arbitrary video sequence. The estimated intrinsics can then be utilized in Structure-from-Motion or SLAM systems.

The method was evaluated in 7 scenes from the [ARKitScenes](https://github.com/apple/ARKitScenes) dataset and 2 scenes from the [BundleFusion](https://graphics.stanford.edu/projects/bundlefusion/) dataset achieving **3.67% and 3.32% Mean Absolute Error** compared to the ground truth values, respectively.


## Method
The method is based on keypoint detection and matching between sequential image frames, that are then used for calculating the fundamental matrix between the images. The fundamental matrix is then decomposed to get the focal lengths in x and y dimensions. More specifically:

1. Keypoint detection and matching is performed using [SuperPoint](https://arxiv.org/abs/1712.07629) + [LightGlue](https://github.com/cvg/LightGlue).

2. The matched keypoints aret then used for estimating the fundamental matric between the 2 views using the robust [Graph-Cut RANSAC](https://github.com/danini/graph-cut-ransac) method.

3. The fundamental matrix is decomposed to its focal length values.

4. We take the median value between the focal lenghts estimated for the different pairs of images, as the final output.


## Setup 
The project depends on [LightGlue](https://github.com/cvg/LightGlue), which we will install based on the official instructions.

1. Create a conda environment

```
conda create --name focal_length_estimation python=3.9.18
conda activate focal_length_estimation
```

2. Clone and install LightGlue from the root directory:

```
git clone https://github.com/cvg/LightGlue.git
cd LightGlue
python -m pip install -e .
cd ..
```

3. Install additional dependencies

```
pip install tqdm
```

## Run it on your images
From the root directory of the project run:

```
python estimate_focal_lengths.py --input_dir <IMAGE_FOLDER_PATH>
```

where `IMAGE_FOLDER_PATH` is the directory where the images of the scene are. The images have to be sequential and in increasing order of names (e.g. first image frame_0001.jpg, second frame_0002.jpg etc.)

## Evaluate on the provided scenes from the ARKitScenes and BundleFusion datasets

### 1. Evaluation in ARKitScenes

Download the data following the official instructions from [here](https://github.com/apple/ARKitScenes/blob/main/DATA.md). The scenes tested are from the 3dod dataset. You need to create a directory called arkitscenes with the following structure, at the root directory of that project. When dowloading the data there will be some extra subfolders which you can simply remove  The directory should have the following format:

```
arkitscenes
      ├── 40753679                           
      ├── 40777060                     
      ├── 40809740                           
      ├── 47331040                 
      └── 47331127 
      ├── 47332493                 
      └── 47333687           
```

From the root directory of the project run:

```
python evaluate_estimation.py --dataset_dir ./arkitscenes --eval_dataset arkitscenes # for arkitscenes
```
or
```
python evaluate_estimation.py --dataset_dir ./bundlefusion --eval_dataset bundlefusion # for bundlefusion

```