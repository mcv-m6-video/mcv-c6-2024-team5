import cv2
import numpy as np
import json
import source.global_variables as gv

## LOADING AND SAVING FUNCTIONS

def load_video():
    cap = cv2.VideoCapture(gv.PATH_TO_VIDEO)
    if not cap.isOpened():
        raise ValueError("Error opening video file")
    return cap

def load_frame_dict():
    with open(f"{gv.PATH_TO_OUTPUT}frame_dict.json") as file:
        frame_dict = json.load(file)
    return frame_dict

def load_mean_std(start=True):
    phase_tag = "start" if start else "end"
    mean = np.load(f"{gv.PATH_TO_TMP}mean_{phase_tag}_{gv.Params.COLOR_TAG}.npy")
    std = np.load(f"{gv.PATH_TO_TMP}std_{phase_tag}_{gv.Params.COLOR_TAG}.npy")
    return mean, std

## SAVING FUNCTIONS

def save_mean_std(mean, std, start=True):
    phase_tag = "start" if start else "end"
    np.save(f"{gv.PATH_TO_TMP}mean_{phase_tag}_{gv.Params.COLOR_TAG}.npy", mean)
    np.save(f"{gv.PATH_TO_TMP}std_{phase_tag}_{gv.Params.COLOR_TAG}.npy", std)

def save_visualizations(mean_to_viz, std_to_viz, start=True):
    phase_tag = "start" if start else "end"
    cv2.imwrite(f"{gv.PATH_TO_OUTPUT}mean_{phase_tag}_{gv.Params.COLOR_TAG}_{gv.Params.MODELLING_TAG}.png", mean_to_viz)
    cv2.imwrite(f"{gv.PATH_TO_OUTPUT}std_{phase_tag}_{gv.Params.COLOR_TAG}_{gv.Params.MODELLING_TAG}.png", std_to_viz)