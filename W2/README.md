# C6 Team 5 - Week 2

## Data
Add the data to the a folder named `data` in the root of the project. The data should be organized as follows:

```
data
│
└───AICity_data
|   |
|   └───train
|   |   └───S03
|   |   |   └───c010
|   |   |   |   └───vdo.avi
└───ai_challenge_s03_c010-full_annotation.xml
```
## Preparation
To install the required packages to run the program, execute the following command:

```bash
pip install -r requirements3_10.txt
```
* We reccomend to use Python 3.10 to run the program.
* We recomend to use a virtual environment to avoid conflicts with other projects.

Also run the following command to create the frame_dict.json file, containing the labels in a more convenient format:
```bash
cd scripts
python proc_xml_to_json.py
```

## Execution
### Training and tuning
To train the model and tune the hyperparameters, execute one of the following commands: 

```bash
cd scripts
python yolo_train.py
python yolo_tune.py
```
To specify the dataset configuration that YOLO usually needs, refer to the [dataset.yaml](scripts/dataset.yaml) file. 

### Object and tracking
To compute the pipeline for the object tracking just execute the following command to run the program choosing the desired tracking method:

```bash
python main.py [--tag TAG][--tracking-method {overlap,kalman_sort}] [--show-tracking] [--save-for-track-eval][--frames-percentage PERCENTAGE]
```

With the following options:\
  --tag TAG             Tag for the output folder\
  --tracking-method     Choose the tracking method
  --show-tracking       If specified, show the tracking results\
  --save-for-track-eval If specified, save the tracking results in a format that can be used for the tracking evaluation\
  --frames-percentage   Percentage of frames to skip while showing the video of the tracking results

Also, indicate paths for model to use in file of [global variables](source/global_variables.py).