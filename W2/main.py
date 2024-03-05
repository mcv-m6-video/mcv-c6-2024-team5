import numpy as np
import cv2
from source.data_io import load_video, load_frame_dict, load_mean_std, save_mean_std, save_visualizations, gt_bboxes, calculate_mAP, init_output_folder, save_metrics, save_gif, save_frames
from source.image_processing import process_frames, compute_mean_std, truncate_values, generate_binary_frames, predict_bboxes
from source.visualization import show_binary_frames, show_frame_with_pred
from source.metrics import compute_video_ap
from source.tracking import overlap_tracking, tracking_kalman, show_tracking
import source.global_variables as gv
from sort.sort import Sort

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
    
    # Check lengths
    print("Length of binary frames:", len(binary_frames))
    print("Length of preds:", len(preds))
    
    # Call the appropriate tracking function
    if gv.Params.TRACKING_METHOD == 'overlap':
        ids_tracking = overlap_tracking(preds)
        show_tracking(cap, binary_frames, ids_tracking, total_frames, gt, preds, aps, map)
    elif gv.Params.TRACKING_METHOD == 'kalman':
        tracked_objects, num_trackers = tracking_kalman(cap, binary_frames, preds, total_frames, frame_dict)
        print("Number of trackers:", num_trackers)
    else:
        print("Unknown tracking method specified.")

    gt = gt_bboxes(frame_dict, total_frames)
    aps = compute_video_ap(gt, preds)
    map = calculate_mAP(gt, preds, aps)
    print("mAP of the video:", map)
    save_metrics(aps, map)
    #save_gif(cap, binary_frames, 250, total_frames, gt, preds, aps, map)
    #save_frames(cap, binary_frames, 250, total_frames, gt, preds, aps, map)
    
    #if gv.Params.SHOW_BINARY_FRAMES:
        #show_frame_with_pred(cap, binary_frames, total_frames, gt, preds, aps, map)
        #show_binary_frames(binary_frames, total_frames, gt, preds, aps, map)
        #show_tracking(cap, binary_frames,ids_tracking, total_frames, gt, preds, aps, map)
    
    # If adaptive modelling is enabled, save the mean and std for the end of the video
    if gv.Params.ADAPTIVE_MODELLING:
        mean_to_viz, std_to_viz = truncate_values(mean, std)
        # show_frame_with_pred(cap, binary_frames, total_frames, gt, preds, aps, map)
        save_visualizations(mean_to_viz, std_to_viz, start=False)


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
