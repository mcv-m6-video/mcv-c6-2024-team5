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

To execute the speed estimation, the velocity.py file should be executed:

```bash
python velocity.py 
```

Make sure you have the video files and the pretrained YOLO model in the same directory as velocity.py

## Multi-camera tracking

### Training the Triplet Loss model for feature extraction
First, generate the triplets for the training of the Triplet Loss model. To do this, the build_triplets.py file should be executed:

```bash
python build_triplets.py
```
That will generate 1200 triplets for each sequence in the dataset. The triplets will be stored in the data folder at /triplets and they will be combined; it will combine S03 and S04 at /S0304 to train a model for S01, S01 and S04 at /S0104 to train a model for S03, and S01 and S03 at /S0103 to train a model for S04.

After generating the triplets, the siamese_network.py file should be executed to train the Triplet Loss model, here one run example (do not add parameters if you want to use the default ones):

```bash
python script_name.py --mode train --combination 0 --lr 0.001 --weight_decay 0.0001 --batch_size 16 --epochs 20
```
Where:
- `--mode train`: Sets the script to training mode.
- `--combination`: Selects the first sequence combination for training. Index for selecting sequence. 0: S0103 data for S04, 1: S0104 for S03, 2: S0304 for S01.
- `--data_path`: Specifies the base directory for the dataset.
- `--models_path`: Specifies the directory to save trained models.
- `--lr`: Sets the learning rate.
- `--weight_decay`: Sets the weight decay for regularization.
- `--batch_size`: Sets the batch size for training.
- `--epochs`: Sets the number of epochs for training.
- `--margin`: Sets the margin for the Triplet Loss function.

For testing the model, the siamese_network.py file should be executed with the following arguments:

```bash
python script_name.py --mode test --combination 0 --batch_size 16
```
Where you can compute the mean distance between the anchor and the positive samples and the mean distance between the anchor and the negative samples for a test set.

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


