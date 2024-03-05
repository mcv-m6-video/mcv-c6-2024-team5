import argparse

# Define Params outside of init() function
class Params:
    RECOMPUTE_MEAN_STD = False
    FRAMES_PERCENTAGE = 0.25
    ALPHA = 4.58
    ADAPTIVE_MODELLING = True
    RHO = 0.06
    COLOR = True
    COLOR_SPACE = "rgb"
    SHOW_BINARY_FRAMES = True
    STATE_OF_THE_ART = True
    TAG = ""
    FRAME_TO_ANALYZE = 550
    PATH_RUN = ""
    TRACKING_METHOD = "overlap" 

def log_params():
    print("--- PARAMETERS ---")
    for key, value in vars(Params).items():
        # Filter out private variables
        if key.startswith("__"):
            continue
        print(f"{key}: {value}")
    print("------------------")

def params_as_dict():
    return {key: value for key, value in vars(Params).items() if not key.startswith("__")}

def init():
    global PATH_TO_DATA, PATH_TO_VIDEO, PATH_TO_TMP, PATH_TO_OUTPUT
    global Params

    # STATIC
    PATH_TO_DATA = '../W1/data/AICity_data/'
    PATH_TO_VIDEO = f"{PATH_TO_DATA}train/S03/c010/vdo.avi"
    PATH_TO_TMP = "./tmp/"
    PATH_TO_OUTPUT = "./output/"

    parser = argparse.ArgumentParser(description='C6 Team 5 - Week 1')
    parser.add_argument('--recompute-mean-std', action='store_true', default=False, help='Whether the mean and standard deviation should be recomputed')
    parser.add_argument('--frames-percentage', type=float, default=0.25, help='Percentage of frames to use for the mean and standard deviation computation')
    parser.add_argument('--alpha', type=float, default=4.58, help='Alpha value for the binary frames computation')
    parser.add_argument('--adaptive-modelling', action='store_true', default=True, help='Whether to use adaptive modelling')
    parser.add_argument('--rho', type=float, default=0.06, help='Rho value for the binary frames computation')
    parser.add_argument('--color', action='store_true', default=True, help='Whether to use color for the binary frames computation')
    parser.add_argument('--color-space', type=str, default="rgb", choices=["rgb", "hsv", "yuv", "lab", "ycrcb"], help="Color space to use for the binary frames computation")
    parser.add_argument('--show-binary-frames', action='store_true', default=True, help='Whether to show the binary frames')
    parser.add_argument('--state-of-the-art', action='store_true', default=True, help="State of the art background subtraction method")
    parser.add_argument('--tag', type=str, default="", help='Tag for the output folder')
    parser.add_argument('--tracking-method', type=str, default="overlap", choices=["overlap", "kalman"], help="Choose the tracking method")
    args = parser.parse_args()
    
    # Update Params values
    Params.RECOMPUTE_MEAN_STD = args.recompute_mean_std
    Params.FRAMES_PERCENTAGE = args.frames_percentage
    Params.ALPHA = args.alpha
    Params.ADAPTIVE_MODELLING = args.adaptive_modelling
    Params.MODELLING_TAG = "adaptive" if Params.ADAPTIVE_MODELLING else "static"
    Params.RHO = args.rho
    Params.COLOR = args.color
    Params.COLOR_TAG = "color" if Params.COLOR else "grayscale"
    Params.COLOR_SPACE = args.color_space if Params.COLOR else ""
    Params.SHOW_BINARY_FRAMES = args.show_binary_frames
    Params.STATE_OF_THE_ART = args.state_of_the_art
    Params.TAG = args.tag
    Params.TRACKING_METHOD = args.tracking_method

    if Params.ADAPTIVE_MODELLING:
        if Params.COLOR:
            PATH_RUN = f"{PATH_TO_OUTPUT}{Params.TAG}_{Params.MODELLING_TAG}_{Params.COLOR_TAG}_{Params.COLOR_SPACE}_alpha={str(Params.ALPHA)}_rho={str(Params.RHO)}/"
        else:
            PATH_RUN = f"{PATH_TO_OUTPUT}{Params.TAG}_{Params.MODELLING_TAG}_{Params.COLOR_TAG}_alpha={str(Params.ALPHA)}_rho={str(Params.RHO)}/"
    else:
        if Params.COLOR:
            PATH_RUN = f"{PATH_TO_OUTPUT}{Params.TAG}_{Params.MODELLING_TAG}_{Params.COLOR_TAG}_{Params.COLOR_SPACE}_alpha={str(Params.ALPHA)}/"
        else:
            PATH_RUN = f"{PATH_TO_OUTPUT}{Params.TAG}_{Params.MODELLING_TAG}_{Params.COLOR_TAG}_alpha={str(Params.ALPHA)}/"

    if Params.STATE_OF_THE_ART:
        PATH_RUN = f"{PATH_TO_OUTPUT}{Params.TAG}_state_of_the_art/"
    
    log_params()
