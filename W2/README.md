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
To compute the pipeline for the object tracking just execute the following command to run the program choosing the desired tracking method:

```bash
python main.py [--tag TAG][--tracking-method {overlap,kalman_sort}]
```

With the following options:\
  --tag TAG             Tag for the output folder\
  --tracking-method     Choose the tracking method
