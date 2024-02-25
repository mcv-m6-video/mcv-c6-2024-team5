import cv2
import numpy as np
import json

## LOADING AND SAVING FUNCTIONS

def load_video(path_to_video):
    cap = cv2.VideoCapture(path_to_video)
    if not cap.isOpened():
        raise ValueError("Error opening video file")
    return cap

def load_frame_dict(path_to_output):
    with open(f"{path_to_output}frame_dict.json") as file:
        frame_dict = json.load(file)
    return frame_dict

def load_mean_std(path_to_tmp):
    mean = np.load(f"{path_to_tmp}mean.npy")
    std = np.load(f"{path_to_tmp}std.npy")
    return mean, std

## SAVING FUNCTIONS

def save_mean_std(mean, std, path_to_tmp):
    np.save(f"{path_to_tmp}mean.npy", mean)
    np.save(f"{path_to_tmp}std.npy", std)

def save_visualizations(mean_to_viz, std_to_viz, path_to_output):
    cv2.imwrite(f"{path_to_output}mean.png", mean_to_viz)
    cv2.imwrite(f"{path_to_output}std.png", std_to_viz)