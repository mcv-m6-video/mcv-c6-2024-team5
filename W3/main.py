import cv2
import time
from source.data_io import load_video, load_frame_dict, gt_bboxes, calculate_mAP, init_output_folder, save_gif, save_for_track_eval
from source.metrics import compute_video_ap
from source.tracking import overlap_tracking, tracking_kalman_sort, overlap_plus_of_tracking
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
    preds_with_ids = []
    t0 = time.time()
    if gv.Params.TRACKING_METHOD == 'overlap':
        preds_with_ids = overlap_tracking(preds)
    elif gv.Params.TRACKING_METHOD == 'kalman_sort':
        preds_with_ids = tracking_kalman_sort(preds)
    elif gv.Params.TRACKING_METHOD == 'overlap_plus_of':
        preds_with_ids = overlap_tracking(preds, use_of=True)
    t1 = time.time()
    print(f"Tracking took {t1 - t0} seconds")

    if gv.Params.SHOW_TRACKING:
        if len(preds_with_ids) > 0:
            show_tracking(cap, total_frames, gt, preds_with_ids, aps, map)

    # save_gif(cap, 250, total_frames, gt, new_preds, aps, map)

    if gv.Params.SAVE_FOR_TRACK_EVAL:
        save_for_track_eval(preds_with_ids)


if __name__ == "__main__":
    main()
