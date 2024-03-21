# C6 Team 5 - Week 4

## Directory structure
The Week 4 directory should have the following structure:

```
data
│
└───aic19-track1-mtmc-train
│
models
│
└───feature_extraction
└───object_detection
```
In the data folder, the aic19-track1-mtmc-train dataset should be stored. The models folder should contain the models for feature extraction and object detection.


## Measuring velocity 

TO COMPLETE

## Multi-camera tracking

### Training the Triplet Loss model for feature extraction
TO COMPLETE

### Performing multi-camera tracking

To execute the multi-camera tracking, the main.py file should be executed with the following arguments:

```bash
python main.py --config path_to_config
```

The path_to_config should be the path to the configuration file that contains the parameters for the multi-camera tracking. The configuration file should be a .yaml file with the following structure:

```yaml
sequence: path_to_sequence
od_model: path_to_object_detection_model
feature_model: path_to_feature_extraction_model
```

The sequence parameter should be the path to the sequence that will be used for the multi-camera tracking. The od_model parameter should be the path to the object detection model that will be used for the multi-camera tracking. The feature_model parameter should be the path to the feature extraction model that will be used for the multi-camera tracking.


