import cv2

from source.data_io import load_video, load_frame_dict, load_mean_std, save_mean_std, save_visualizations
from source.image_processing import process_frames, compute_mean_std, truncate_values, generate_binary_frames
from source.visualization import show_binary_frames
import source.global_variables as gv

def main():
    gv.init()
    cap = load_video()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_dict = load_frame_dict()
    
    if not gv.Params.MEAN_STD_COMPUTED:
        frames = process_frames(cap, total_frames)
        mean, std = compute_mean_std(frames)
        save_mean_std(mean, std)
    else:
        mean, std = load_mean_std()
    
    mean_to_viz, std_to_viz = truncate_values(mean, std)
    save_visualizations(mean_to_viz, std_to_viz)
    
    binary_frames = generate_binary_frames(cap, total_frames, mean, std)
    show_binary_frames(binary_frames, frame_dict, total_frames)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
