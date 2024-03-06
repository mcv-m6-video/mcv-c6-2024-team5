import cv2
from source.data_io import load_video, load_frame_dict, gt_bboxes, calculate_mAP, init_output_folder, save_gif
from source.metrics import compute_video_ap
from source.tracking import overlap_tracking, tracking_kalman_sort
import source.global_variables as gv
from source.inference import predict
from source.visualization import show_tracking


def main():
    # Initialize global variables and output folder
    gv.init()
    init_output_folder()
    cap = load_video()
    # Do predictions
    preds = predict()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_dict = load_frame_dict()
    gt = gt_bboxes(frame_dict, total_frames)
    aps = compute_video_ap(gt, preds)
    map = calculate_mAP(gt, preds, aps)

    # Call the appropriate tracking function
    if gv.Params.TRACKING_METHOD == 'overlap':
        preds_with_ids = overlap_tracking(preds)
        show_tracking(cap, preds_with_ids, total_frames, gt, aps, map)
    elif gv.Params.TRACKING_METHOD == 'kalman_sort':
        preds_with_ids = tracking_kalman_sort(preds)
        show_tracking(cap, preds_with_ids, total_frames, gt, aps, map)

    # save_gif(cap, 250, total_frames, gt, new_preds, aps, map)


if __name__ == "__main__":
    main()
