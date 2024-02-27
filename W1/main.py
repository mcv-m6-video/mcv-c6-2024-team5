import cv2

from source.data_io import load_video, load_frame_dict, load_mean_std, save_mean_std, save_visualizations, gt_bboxes, calculate_mAP, init_output_folder, save_metrics
from source.image_processing import process_frames, compute_mean_std, truncate_values, generate_binary_frames, predict_bboxes
from source.visualization import show_binary_frames, show_frame_with_pred
from source.metrics import compute_video_ap
import source.global_variables as gv

def main():
    # Initialize global variables and output folder
    gv.init()
    init_output_folder()

    cap = load_video()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_dict = load_frame_dict()
    
    if gv.Params.RECOMPUTE_MEAN_STD:
        frames = process_frames(cap, total_frames)
        mean, std = compute_mean_std(frames)
        save_mean_std(mean, std)
    else:
        try:
            mean, std = load_mean_std()
        except FileNotFoundError:
            # Log error and exit
            print("Error: mean and std files not found")
            return
    
    mean_to_viz, std_to_viz = truncate_values(mean, std)
    save_visualizations(mean_to_viz, std_to_viz)
    
    binary_frames = generate_binary_frames(cap, total_frames, mean, std)
    preds = predict_bboxes(binary_frames)
    gt = gt_bboxes(frame_dict, total_frames)
    aps = compute_video_ap(gt, preds)
    map = calculate_mAP(gt, preds, aps)
    save_metrics(aps, map)
    if gv.Params.SHOW_BINARY_FRAMES:
        show_frame_with_pred(cap, binary_frames, total_frames, gt, preds, aps, map)
        #show_binary_frames(binary_frames, total_frames, gt, preds, aps, map)
    
    # If adaptive modelling is enabled, save the mean and std for the end of the video
    if gv.Params.ADAPTIVE_MODELLING:
        mean_to_viz, std_to_viz = truncate_values(mean, std)
        # show_frame_with_pred(cap, binary_frames, total_frames, gt, preds, aps, map)
        save_visualizations(mean_to_viz, std_to_viz, start=False)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
