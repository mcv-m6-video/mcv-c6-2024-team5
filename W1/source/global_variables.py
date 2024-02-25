import argparse

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
    PATH_TO_DATA = './data/AICity_data/'
    PATH_TO_VIDEO = f"{PATH_TO_DATA}train/S03/c010/vdo.avi"
    PATH_TO_TMP = "./tmp/"
    PATH_TO_OUTPUT = "./output/"

    parser = argparse.ArgumentParser(description='C6 Team 5 - Week 1')
    parser.add_argument('--recompute-mean-std', action='store_true', default=False, help='Whether the mean and standard deviation should be recomputed')
    parser.add_argument('--frames-percentage', type=float, default=0.25, help='Percentage of frames to use for the mean and standard deviation computation')
    parser.add_argument('--alpha', type=float, default=10, help='Alpha value for the binary frames computation')
    parser.add_argument('--adaptive-modelling', action='store_true', default=True, help='Whether to use adaptive modelling')
    parser.add_argument('--rho', type=float, default=0.4, help='Rho value for the binary frames computation')
    parser.add_argument('--rgb-color', action='store_true', default=False, help='Whether to use RGB color for the binary frames computation')
    parser.add_argument('--show-binary-frames', action='store_true', default=True, help='Whether to show the binary frames')
    args = parser.parse_args()
    
    # PARAMETERS
    class Params:
        RECOMPUTE_MEAN_STD = args.recompute_mean_std
        FRAMES_PERCENTAGE = args.frames_percentage
        ALPHA = args.alpha
        ADAPTIVE_MODELLING = args.adaptive_modelling
        MODELLING_TAG = "adaptive" if ADAPTIVE_MODELLING else "static"
        RHO = args.rho
        RGB_COLOR = args.rgb_color
        COLOR_TAG = "rgb" if RGB_COLOR else "grayscale"
        SHOW_BINARY_FRAMES = args.show_binary_frames
    
    log_params()