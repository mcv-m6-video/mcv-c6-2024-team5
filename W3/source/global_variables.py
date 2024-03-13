import argparse

# Define Params outside of init() function
class Params:
    TAG = ""
    PATH_RUN = ""
    TRACKING_METHOD = "overlap"
    SHOW_TRACKING = False
    SAVE_FOR_TRACK_EVAL = False

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
    global PATH_TO_DATA, PATH_TO_VIDEO, PATH_TO_TMP, PATH_TO_OUTPUT, PATH_TO_MODEL, PATH_TO_OF
    global SEQ, CAM
    global Params

    # STATIC
    PATH_TO_DATA = 'scripts/aic19-track1-mtmc-train/train/'
    SEQ = "S03"
    CAM = "c010"
    PATH_TO_VIDEO = f"{PATH_TO_DATA}{SEQ}/{CAM}/vdo.avi"
    PATH_TO_OUTPUT = "./output/"
    PATH_TO_MODEL = "./best_all.pt"  # "f"scripts/runs/detect/train_{SEQ}/weights/best.pt"
    PATH_TO_OF = "./W2_video_of"

    parser = argparse.ArgumentParser(description='C6 Team 5 - Week 2')
    parser.add_argument('--tag', type=str, default="", help='Tag for the output folder')
    parser.add_argument('--tracking-method', type=str, default="overlap_plus_of", choices=["overlap", "kalman_sort", "overlap_plus_of"], help="Choose the tracking method")
    parser.add_argument('--show-tracking', action="store_true", help="Show the tracking")
    parser.add_argument('--save-for-track-eval', action="store_true", help="Save the tracking for evaluation")
    parser.add_argument('--frames-percentage', type=float, default=0.25, help="Amount of frames to skip when doing the tracking evaluation. Default is 0.25 (25%)")
    args = parser.parse_args()
    
    # Update Params values
    Params.TRACKING_METHOD = args.tracking_method
    Params.PATH_RUN = f"{PATH_TO_OUTPUT}{Params.TAG}{Params.TRACKING_METHOD}/"
    Params.SHOW_TRACKING = False
    Params.SAVE_FOR_TRACK_EVAL = False
    Params.FRAMES_PERCENTAGE = args.frames_percentage
    Params.CAM = CAM
    log_params()
