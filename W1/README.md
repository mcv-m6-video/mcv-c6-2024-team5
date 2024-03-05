# C6 Team 5 - Week 1

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
To compute the pipeline to compute the background modelling just execute the following command to run the program:

```bash
python main.py [--recompute-mean-std] [--frames-percentage FRAMES_PERCENTAGE] [--alpha ALPHA] [--adaptive-modelling] [--rho RHO] [--color] [--color-space {rgb,hsv,yuv,lab,ycrcb}] [--show-binary-frames] [--state-of-the-art] [--tag TAG]
```

With the following options:\
  --recompute-mean-std  Whether the mean and standard deviation should be recomputed\
  --frames-percentage   Percentage of frames to use for the mean and standard deviation computation\
  --alpha ALPHA         Alpha value for the binary frames computation\
  --adaptive-modelling  Whether to use adaptive modelling\
  --rho RHO             Rho value for the binary frames computation\
  --color               Whether to use color for the binary frames computation\
  --color-space         Color space to use for the binary frames computation  |  Options:{rgb,hsv,yuv,lab,ycrcb}\
  --show-binary-frames  Whether to show the binary frames\
  --state-of-the-art    State of the art background subtraction method\
  --tag TAG             Tag for the output folder