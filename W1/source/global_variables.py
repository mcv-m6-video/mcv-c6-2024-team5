import argparse

def init():
    global PATH_TO_DATA, PATH_TO_VIDEO, PATH_TO_TMP, PATH_TO_OUTPUT
    global Params

    # STATIC
    PATH_TO_DATA = './data/AICity_data/'
    PATH_TO_VIDEO = f"{PATH_TO_DATA}train/S03/c010/vdo.avi"
    PATH_TO_TMP = "./tmp/"
    PATH_TO_OUTPUT = "./output/"

    parser = argparse.ArgumentParser(description='C6 Team 5 - Week 1')
    parser.add_argument('--mean-std-computed', action='store_true', default=False, help='Whether the mean and standard deviation have been computed')
    parser.add_argument('--frames-percentage', type=float, default=0.25, help='Percentage of frames to use for the mean and standard deviation computation')
    parser.add_argument('--alpha', type=int, default=10, help='Alpha value for the binary frames computation')
    args = parser.parse_args()
    
    # PARAMETERS
    class Params:
        MEAN_STD_COMPUTED = args.mean_std_computed
        FRAMES_PERCENTAGE = args.frames_percentage
        ALPHA = args.alpha
        