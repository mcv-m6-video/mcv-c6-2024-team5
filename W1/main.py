import cv2

from source.data_io import load_video, load_frame_dict, load_mean_std, save_mean_std, save_visualizations
from source.image_processing import process_frames, compute_mean_std, truncate_values, generate_binary_frames
from source.visualization import show_binary_frames

## MAIN AND CONSTANTS

PATH_TO_DATA = './data/AICity_data/'
PATH_TO_VIDEO = f"{PATH_TO_DATA}train/S03/c010/vdo.avi"
PATH_TO_TMP = "./tmp/"
PATH_TO_OUTPUT = "./output/"

MEAN_STD_COMPUTED = True
FRAMES_PERCENTAGE = 0.25
ALPHA = 10

def main():
    cap = load_video(PATH_TO_VIDEO)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_dict = load_frame_dict(PATH_TO_OUTPUT)
    
    if not MEAN_STD_COMPUTED:
        frames = process_frames(cap, total_frames, FRAMES_PERCENTAGE)
        mean, std = compute_mean_std(frames)
        save_mean_std(mean, std, PATH_TO_TMP)
    else:
        mean, std = load_mean_std(PATH_TO_TMP)
    
    mean_to_viz, std_to_viz = truncate_values(mean, std)
    save_visualizations(mean_to_viz, std_to_viz, PATH_TO_OUTPUT)
    
    binary_frames = generate_binary_frames(cap, total_frames, FRAMES_PERCENTAGE, mean, std, ALPHA)
    show_binary_frames(binary_frames, frame_dict, total_frames, FRAMES_PERCENTAGE)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
