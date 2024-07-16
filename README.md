# PEGG-Net
This repository contains the official implementation of PEGG-Net from the paper:

**PEGG-Net: Pixel-Wise Efficient Grasp Generation in Complex Scenes**

Haozhe Wang, Zhiyang Liu, Lei Zhou, Huan Yin and Marcelo H. Ang Jr.

2024 IEEE International Conference on Cybernetics and Intelligent Systems (CIS) and IEEE Conference on Robotics, Automation and Mechatronics (RAM)

[[Paper](https://arxiv.org/pdf/2203.16301.pdf)][[Demo Video](https://www.youtube.com/watch?v=wsDUP60PC1E)]

Please clone this GitHub repo before proceeding with the installation.

```bash
git clone https://github.com/HZWang96/PEGG-Net.git
```

## Installation using Anaconda

The code was tested on Ubuntu 18.04, with Python 3.6 and PyTorch 1.7.0 (CUDA 11.0). NVIDIA GPUs are needed for both training and testing.

1. Create a new conda environment
    
    ```bash
    conda create --name peggnet python=3.6
    ```
    
2. Install PyTorch 1.7.0 for CUDA 11.0
    
    ```bash
    conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
    ```
    
3. Install the required Python packages
    
    ```bash
    pip install -r requirements.txt
    ```
    

## Installation using Docker

1. Install the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html)
2. Pull the PyTorch 1.7.0 docker image from [docker hub](https://hub.docker.com/r/pytorch/pytorch/tags)
    
    ```bash
    docker pull pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
    ```
    
3. Run the following command to start the docker container
    
    ```bash
    nvidia-docker run --gpus all --ipc host -it -v <path/to/local/directory>:<workspace/in/docker/container> pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel bash
    ```
    
4. Configure the docker container by running the following commands:
    
    ```bash
    chmod 755 docker_config
    ./docker_config
    ```
    

## Dataset Preparation

1. Download and extract the [Cornell Grasping Dataset](https://www.kaggle.com/oneoneliu/cornell-grasp).
2. Download and extract the [Jacquard Dataset](https://jacquard.liris.cnrs.fr/files/database/download.php).
3. For the Cornell and Jacquard dataset, the folders containing the images and labels should be arranged in the following manner:
    
    ```
    PEGG-Net
    | - - data
     `- - | - - cornell
          |  `- - | - - 01
          |       | - - 02
          |       | - - 03
          |       | - - 04
          |       | - - 05
          |       | - - 06
          |       | - - 07
          |       | - - 08
          |       | - - 09
          |       | - - 10
          |        ` - - backgrounds
          ` - - jacquard
            ` - - | - - Jacquard_Dataset_0
                  | - - Jacquard_Dataset_1
                  | - - Jacquard_Dataset_2
                  | - - Jacquard_Dataset_3
                  | - - Jacquard_Dataset_4
                  | - - Jacquard_Dataset_5
                  | - - Jacquard_Dataset_6
                  | - - Jacquard_Dataset_7
                  | - - Jacquard_Dataset_8
                  | - - Jacquard_Dataset_9
                  | - - Jacquard_Dataset_10
                   ` - - Jacquard_Dataset_11
    ```
    
4. For the Cornell Grasping Dataset. convert the PCD files (pcdXXXX.txt) to depth images by running
    
    ```bash
    python -m utils.dataset_preprocessing.generate_cornell_depth data/cornell
    ```
    

## Training

Run `train.py --help` to see the full list of options and description for each option.

.A basic example would be:

```bash
python train.py --description <write a description> --network peggnet --dataset cornell --dataset-path data/cornell --use-rgb 1 --use-depth 0
```

For training on an image-wise split using the Cornell dataset:

```bash
python train.py --description peggnet_iw_rgb_304 --network peggnet --dataset cornell --dataset-path data/cornell --image-wise --use-depth 0 --use-rgb 1 --num-workers 4 --input-size 304
```

Some important flags are:

- `--dataset` to select the dataset you want to use for training.
- `--dataset-path` to provide the path to the selected dataset.
- `--random-seed` to train the network using an image-wise split.
- `--augment` to use random rotations and zooms to augment the dataset.
- `--input-size` to change the size of the input image. **Note that the input image must be a multiple of 8**
- `--use-rgb` to use RGB images during training. Set 1 for true and 0 for false.
- `--use-depth` to use depth images during training.  Set 1 for true and 0 for false.

To train on the Cornell Grasping Dataset using only RGB or depth images, you can use the default hyperparameters and include the `--augment` flag.

For training on the Cornell Grasping Dataset using the image-wise split, add the `--image-wise` flag. The random seed (`--random-seed`) used for shuffling the dataset is 10.

When training using the Jacquard dataset, do not use the `--augment` flag.

To train on the Jacquard Grasping Dataset using only RGB or depth images, you can use the default hyperparameters without the `--augment` flag.

To train on the Cornell Grasping Dataset or the Jacquard dataset using RGB-D images, change the following hyperparameters:

1. Set `--lr 0.01`
2. Set `--lr-step 25,40` 

The trained models will be stored in the `output/models` directory. The TensorBoard log files for each training session will be stored in the `tensorboard` directory.

## Evaluation/Visualization

Run `eval.py --help` to see the full list of options and description for each option.

Some important flags are:

- `--iou-eval` to evaluate using the IoU between grasping rectangles metric
- `--jacquard-ouptut` to generate output files in the format required for simulated testing against the Jacquard dataset.
- `--vis` to plot the network output and predicted grasp rectangles

A basic example would be:

```bash
python eval.py --network <path to trained network> --dataset jacquard --dataset-path data/jacquard --jacquard-output --iou-eval
```
## Running the PEGG-Net Grasping System on the Kinova Movo Robot
Connect network of the inference PC to the Movo2 PC and set the Movo2 PC as ROS Master.

Bring up RGB-D aligned realsense camera ROS node:
```
roslaunch realsense2_camera rs_aligned_depth.launch
```

Or bring up realsense camera ROS node for depth-only prediction:
```
roslaunch realsense2_camera rs_camera.launch
```

To publish tf info of right end-effector in right_base_link frame and calibrated camera extrinsics:
```
movo_tf_publisher/right_base_link.py
movo_tf_publisher/camera_calibration.py
```
To implement prediction with RGB-D input and send results to the control system:
```
python pegg_rgbd_prediction.py
```
Or to implement prediction with depth-only input and send results to the control system:
```
python pegg_d_prediction.py
```
To start control system:
```
python pegg_movo_control.py
```
## Citation
If you find our work useful for your research, please consider citing the following BibTeX entry:
```
@misc{wang2023peggnet,
      title={PEGG-Net: Pixel-Wise Efficient Grasp Generation in Complex Scenes}, 
      author={Haozhe Wang and Zhiyang Liu and Lei Zhou and Huan Yin and Marcelo H Ang Jr au2},
      year={2023},
      eprint={2203.16301},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
