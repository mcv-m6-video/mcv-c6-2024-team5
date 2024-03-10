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
    |
    └───image_0
        └───000xxx_{10_11}.png
```
This is, for the moment, the only data that is going to be used from the KITTI dataset. The `{10_11}` is a placeholder for the two digits that represent the frame number. The `xxx` is a placeholder for the three digits that represent the sequence number.
Get the data from the [KITTI website](https://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo) and click on the "Download stero/optical flow data set (2 GB)" link. (Registration required)

## Off the shelf methods
The following methods are going to be used to compute the optical flow:
- [Pyflow](https://github.com/pathak22/pyflow?tab=readme-ov-file), project based ont the [Coarse2Fine Optical Flow](https://people.csail.mit.edu/celiu/OpticalFlow/) method.

### Pyflow
Refer to the [README.md](/pyflow/README.md) file for the instructions.