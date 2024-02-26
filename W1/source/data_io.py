import cv2
import numpy as np
import json
import os

import source.global_variables as gv

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