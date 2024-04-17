# Action classification on HMDB51

## Download the dataset

0. Before downloading any data, install the `unrar` and `ffmpeg` packages with your package manager. In Debian-based distros (e.g., Ubuntu), this is done with the following command: `sudo apt install unrar ffmpeg`.

1. Now, download the videos from the authors' website and decompress them with `unrar`:

    ```bash
        # Download the compressed dataset
        $ wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar
        # Extract the file
        $ unrar x hmdb51_org.rar
        # Rename the decompressed folder from hmdb51_org -> videos
        $ mv hmdb51_org videos
        # Inspect the contents of the uncompressed folder
        $ ls videos
    ```

You should be seeing 51 folders, one for each action category (`brush_hair`, `cartwheel`, etc) containing video data (`.avi` files):

    ```
        videos/
        |____ brush_hair/
        |     |____ April_09_brush_hair_u_nm_np1_ba_goo_0.avi
        |     |____ April_09_brush_hair_u_nm_np1_ba_goo_1.avi
        |     |____ ...
        |
        |____ cartwheel/
        |     |____ ...
        |
        |____ ...
    ```

However, for our training procedure, it'll be much more convenient having the videos converted to frames first. Frames are faster to read than video and allow selective access depending on how we decide to sample clips.

## Data preparation

To extract the frames in a directory named `frames`, we can run this long one-line command:

    ```bash
    # Make sure you are in the same directory that contains the videos/ folder
    $ find videos -name "*.avi" -print0 | xargs -0 -I {} sh -c 'original="{}"; modified=$(echo "$original" | sed -e "s|videos|frames|" | sed -e "s|\.[^.]*$||"); mkdir -p $modified; ffmpeg -i $original -loglevel error $modified/frame%05d.jpg'
    ```
    
If run correctly, such command will create  `frames/` directory with the same structure as `videos/`, but replacing each video file by a directory containing the frames (in .jpg format). It might take from 30' to an hour to extract the frames (depending on your CPU).

Graphically:

    ```
    frames/
    |____ brush_hair/
    |     |____ April_09_brush_hair_u_nm_np1_ba_goo_0/
    |     |     |____ frame00001.jpg
    |     |     |____ frame00002.jpg
    |     |     |____ frame00003.jpg
    |     |     |____ ...
    |     |____ April_09_brush_hair_u_nm_np1_ba_goo_1/
    |           |____ frame00001.jpg
    |           |____ frame00002.jpg
    |           |____ frame00003.jpg
    |           |____ ...
    |
    |____ cartwheel/
    |     |____ ...
    |
    |____ ...
    ```

## Custom groundtruth

Do not download and use the groundtruth annotations from the authors' webpage, as we will be using a modified version that you'll find in `data/hmbd51/testTrainMulti_601030_splits` directory of this same repository.

Differently from the original groundtruth, we will reserve part of the training videos for validation. In particular, instead of having 70% training and 30% testing data, we will have 60% training, 10% validation, and 30% testing.

Then, just take into account that HMDB51 was thought to be evaluated in 3-fold cross validation. So you will see 3 different splits, namely split1, split2, and split3. In the provided annotations, this splits are done in a separate file for each action label (i.e, `<action_label>_test_split<1, 2, or 3>.txt`). However, we will focus on split1 only (ignore split2 and split3 files). Go and examine any `<action_label>_test_split1.txt` file and you'll find there's a line per video. Each line has the video name followed by an integer that represents the partition (train = 1, validation = 3 or test = 2) for this particular split.

## Run the baseline code

### Prerequisites

Before running the script, you need to install the required Python dependencies listed in the `requirements.txt` file. To install these dependencies using PIP, execute:

    ```bash
    $ pip3 install -r requirements.txt
    ```

### Usage

Finally, the model can be run by executing `src/train.py` script, which expects one positional argument (the directory containing the frames that we've created before), but accepts other multiple arguments:

    ```
    $ python3 src/train.py --help
        usage: train.py [-h] [--annotations-dir ANNOTATIONS_DIR] [--clip-length CLIP_LENGTH]
                [--crop-size CROP_SIZE] [--temporal-stride TEMPORAL_STRIDE]
                [--model-name MODEL_NAME] [--load-pretrain] [--optimizer-name OPTIMIZER_NAME]
                [--lr LR] [--epochs EPOCHS] [--batch-size BATCH_SIZE]
                [--batch-size-eval BATCH_SIZE_EVAL] [--validate-every VALIDATE_EVERY]
                [--num-workers NUM_WORKERS] [--device DEVICE] [--early-stopping EARLY_STOPPING]
                [--wandb] [--load-model LOAD_MODEL] [--only-inference]
                [--clips-per-video CLIPS_PER_VIDEO] [--crops-per-clip CROPS_PER_CLIP]
                [--tsn-k TSN_K]
                frames-dir

    Train a video classification model on HMDB51 dataset.

    positional arguments:
    frames_dir              Directory containing video files

    options:
    -h, --help              Show this help message and exit
    --annotations-dir ANNOTATIONS_DIR
                            Directory containing annotation files (default: data/hmdb51/testTrainMulti_601030_splits)
    --clip-length CLIP_LENGTH
                            Number of frames of the clips (default: 4)
    --crop-size CROP_SIZE
                            Size of spatial crops (squares) (default: 182)
    --temporal-stride TEMPORAL_STRIDE
                            Temporal stride to adjust the receptive field (default: 12)
    --model-name MODEL_NAME
                            Model name as defined in models/model_creator.py (default: x3d_xs)
    --load-pretrain         Load pretrained weights for the model (if available)
    --optimizer-name OPTIMIZER_NAME
                            Optimizer name (supported: "adam" and "sgd" for now) (default: adam)
    --lr LR                 Learning rate (default: 1e-4)
    --epochs EPOCHS         Number of epochs (default: 50)
    --batch-size BATCH_SIZE
                            Batch size for the training data loader (default: 16)
    --batch-size-eval BATCH_SIZE_EVAL
                            Batch size for the evaluation data loader (default: 16)
    --validate-every VALIDATE_EVERY
                            Number of epochs after which to validate the model (default: 1)
    --num-workers NUM_WORKERS
                            Number of worker processes for data loading (default: 2)
    --device DEVICE         Device to use for training (cuda or cpu) (default: cuda)
    --early-stopping EARLY_STOPPING
                            Number of epochs to wait after last time validation loss improved (default: 5)
    --wandb                 Use Weights & Biases for logging
    --load-model LOAD_MODEL
                            Load a model from a file (default: None)
    --only-inference        Only perform inference on the test set (requires a model to load)
    --clips-per-video CLIPS_PER_VIDEO
                            Number of clips to sample per video (default: 3)
    --crops-per-clip CROPS_PER_CLIP
                            Number of spatial crops to sample per clip (default: 1)
    --tsn-k TSN_K
                            Number of clips to sample per video for TSN aggregation (default: 3)

    ```

### Example runs

If not specified, default values should allow you to run the script without issues.

#### Training with defaults

```bash
$ python src/train.py data/frames
```

#### Using a specific model and enabling Weights & Biases logging:

```bash
$ python src/train.py --model-name x3d_m --wandb data/frames
```

#### Performing inference only

```bash
$ python src/train.py --only-inference --load-model model.pth data/frames
```