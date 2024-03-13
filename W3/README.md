# C6 Team 5 - Week 3

## Data
Add the KITTI data to the a folder named `data` in the root of the week folder. The data should be organized as follows:

```
data
│
└───training
|   |
|   └───image_0
|   |   └───000xxx_{10_11}.png
|   └───flow_noc
|   |   └───000xxx_{10_11}.png
└───testing
|   |
|   └───image_0
|       └───000xxx_{10_11}.png
└───AICity_data
|   |
|   └───train
|   |   └───S03
|   |   |   └───c010
|   |   |   |   └───vdo.avi
└───ai_challenge_s03_c010-full_annotation.xml
```
This is, for the moment, the only data that is going to be used from the KITTI dataset. The `{10_11}` is a placeholder for the two digits that represent the frame number. The `xxx` is a placeholder for the three digits that represent the sequence number.
Get the data from the [KITTI website](https://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo) and click on the "Download stero/optical flow data set (2 GB)" link. (Registration required)

## Off the shelf methods
The following methods are going to be used to compute the optical flow:
- [Pyflow](https://github.com/pathak22/pyflow?tab=readme-ov-file), project based ont the [Coarse2Fine Optical Flow](https://people.csail.mit.edu/celiu/OpticalFlow/) method.
- [RAFT](https://github.com/princeton-vl/RAFT), a method based on the [RAFT: Recurrent All-Pairs Field Transforms for Optical Flow](https://arxiv.org/abs/2003.12039) paper.
- [Ef-RAFT](https://github.com/n3slami/Ef-RAFT), a method based on the [Rethinking RAFT for Efficient Optical Flow](https://arxiv.org/abs/2401.00833) paper.

### Pyflow
Refer to the [README.md](/pyflow/README.md) file for the instructions.

### RAFT

#### Requirements
The code has been tested with PyTorch 1.6 and Cuda 10.1.
```bash	
conda create --name raft
conda activate raft
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy opencv -c pytorch
```

#### Preparation
Just add the frames to the `data` folder with an ordered naming. 

#### Execution
Execute the following command to install the required dependencies and compute the optical flow:
```bash
cd RAFT
python .\demo.py --model=models/ours-things.pth --path=data
```
Then you will obtain the results of the optical flow, raw and colorized for a better visualization.

### Ef-RAFT

#### Requirements
The code has been tested with PyTorch 1.6 and Cuda 10.1.
```bash	
conda create --name efraft
conda activate raft
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy opencv -c pytorch
```

#### Preparation
Just add the frames to the `data` folder with an ordered naming.

#### Execution
Execute the following command to install the required dependencies and compute the optical flow:
```bash
cd Ef-RAFT
python .\demo.py --model=models/raft-things.pth --path=data
```
Then you will obtain the results of the optical flow, raw and colorized for a better visualization.

## Object tracking with Optical Flow

#### Requirements
The optical flow between each of the frames of the video has to be computed previously. 
The folder where the optical flow is stored has to be indicated in the PATH_TO_OF variable in the global_variables.py file. 


#### Execution

To execute the object tracking with optical flow, the method has been added as one of the possible tracking
methods in the main.py file. To execute it, we just run it as last week's code, but with the --method parameter set to "overlap_plus_of".
```bash
python main.py --method overlap_plus_of
```

The results will be stored in the output folder, in a folder with the same name as the method used.