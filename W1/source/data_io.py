import cv2
import numpy as np
import json
import os

import source.global_variables as gv
from source.metrics import compute_ap

## LOADING FUNCTIONS

def load_video():
    cap = cv2.VideoCapture(gv.PATH_TO_VIDEO)
    if not cap.isOpened():
        raise ValueError("Error opening video file")
    return cap

def load_frame_dict():
    with open(f"frame_dict.json") as file:
        frame_dict = json.load(file)
    return frame_dict

def load_mean_std(start=True):
    phase_tag = "start" if start else "end"
    mean = np.load(f"{gv.PATH_TO_TMP}mean_{phase_tag}_{gv.Params.COLOR_TAG}_alpha{str(gv.Params.ALPHA)}.npy")
    std = np.load(f"{gv.PATH_TO_TMP}std_{phase_tag}_{gv.Params.COLOR_TAG}_alpha{str(gv.Params.ALPHA)}.npy")
    # mean = np.load(f"{gv.PATH_TO_TMP}mean_{phase_tag}_{gv.Params.COLOR_TAG}_alpha10.npy")
    # std = np.load(f"{gv.PATH_TO_TMP}std_{phase_tag}_{gv.Params.COLOR_TAG}_alpha10.npy")
    return mean, std

## SAVING FUNCTIONS

def init_output_folder():
    os.makedirs(f"{gv.Params.PATH_RUN}", exist_ok=True)
    params_dict = gv.params_as_dict()
    with open(f"{gv.Params.PATH_RUN}params.json", "w") as file:
        json.dump(params_dict, file)

def save_mean_std(mean, std, start=True):
    phase_tag = "start" if start else "end"
    np.save(f"{gv.PATH_TO_TMP}mean_{phase_tag}_{gv.Params.COLOR_TAG}_alpha{str(gv.Params.ALPHA)}.npy", mean)
    np.save(f"{gv.PATH_TO_TMP}std_{phase_tag}_{gv.Params.COLOR_TAG}_alpha{str(gv.Params.ALPHA)}.npy", std)

def save_visualizations(mean_to_viz, std_to_viz, start=True):
    phase_tag = "start" if start else "end"
    # Change color space to BGR for visualization
    if gv.Params.COLOR:
        if gv.Params.COLOR_SPACE == "hsv":
            mean_to_viz = cv2.cvtColor(mean_to_viz, cv2.COLOR_HSV2BGR)
            std_to_viz = cv2.cvtColor(std_to_viz, cv2.COLOR_HSV2BGR)
        elif gv.Params.COLOR_SPACE == "yuv":
            mean_to_viz = cv2.cvtColor(mean_to_viz, cv2.COLOR_YUV2BGR)
            std_to_viz = cv2.cvtColor(std_to_viz, cv2.COLOR_YUV2BGR)
        elif gv.Params.COLOR_SPACE == "lab":
            mean_to_viz = cv2.cvtColor(mean_to_viz, cv2.COLOR_LAB2BGR)
            std_to_viz = cv2.cvtColor(std_to_viz, cv2.COLOR_LAB2BGR)
        elif gv.Params.COLOR_SPACE == "ycrcb":
            mean_to_viz = cv2.cvtColor(mean_to_viz, cv2.COLOR_YCrCb2BGR)
            std_to_viz = cv2.cvtColor(std_to_viz, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(f"{gv.Params.PATH_RUN}mean_{phase_tag}.png", mean_to_viz)
    cv2.imwrite(f"{gv.Params.PATH_RUN}std_{phase_tag}.png", std_to_viz)

def gt_bbox(frame_dict, frame_number):
    bboxes = []
    for car_id, car in frame_dict[str(frame_number)].items():
        if car['is_parked']:
            continue
        x1, y1, x2, y2 = int(car['xtl']), int(car['ytl']), int(car['xbr']), int(car['ybr'])
        bboxes.append((x1, y1, x2, y2))
    return bboxes

def gt_bboxes(frame_dict, total_frames):
    gt = []
    for i in range(int(total_frames * gv.Params.FRAMES_PERCENTAGE), total_frames):
        gt.append(gt_bbox(frame_dict, i))
    return gt

def gt_bboxes_comparison(frame_dict, total_frames, percentage_frames):
    """
    Get ground truth bounding boxes for each frame based on the given percentage of frames.
    :param frame_dict: Dictionary containing frame annotations
    :param total_frames: Total number of frames in the video
    :param percentage_frames: Percentage of frames to consider
    :return: List of ground truth bounding boxes for each frame
    """
    gt = []
    frame_indices = sorted([int(i) for i in frame_dict.keys()])

    if percentage_frames < 1.0:
        target_frames = int(total_frames * percentage_frames)
        frame_indices = frame_indices[:target_frames]

    for i in range(1, total_frames + 1):
        if str(i) in frame_dict and i in frame_indices:
            gt.append(gt_bbox(frame_dict, i))
        else:
            gt.append([])  # If frame annotation is missing or not in selected frames, append empty list
    
    return gt

def calculate_mAP(gts, preds, aps):
    valid_aps = []
    for gt, pred, ap in zip(gts, preds, aps):
        if len(gt) > 0 or len(pred) > 0:
            valid_aps.append(ap)
    return np.mean(valid_aps)

def calculate_mAP_comparison(gt_annotations, all_predictions):
    """
    Compute the mean Average Precision (mAP) given ground truth annotations and predicted bounding boxes.
    """
    aps = []
    for gt_boxes, pred_boxes in zip(gt_annotations, all_predictions):
        ap = compute_ap(gt_boxes, pred_boxes)
        aps.append(ap)

    map_value = np.mean(aps)  # Calculate mean AP for all frames (including those with no ground truth)

    return aps, map_value

def save_metrics(aps, map):
    results = {"aps": aps, "mAP of the video": map}
    with open(f"{gv.Params.PATH_RUN}results.json", "w") as file:
        json.dump(results, file)
