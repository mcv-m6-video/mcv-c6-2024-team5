import cv2
import numpy as np
import source.global_variables as gv

## PROCESSING FUNCTIONS

def process_frames(cap, total_frames):
    frames = []
    for i in range(int(total_frames * gv.Params.FRAMES_PERCENTAGE)):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
    return np.array(frames)

def compute_mean_std(frames):
    mean = np.mean(frames, axis=0)
    std = np.std(frames, axis=0)
    return mean, std

def truncate_values(mean, std):
    mean_to_viz = np.clip(mean, 0, 255).astype(np.uint8)
    std_to_viz = np.clip(std, 0, 255).astype(np.uint8)
    return mean_to_viz, std_to_viz

def generate_binary_frames(cap, total_frames, mean, std):
    binary_frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * gv.Params.FRAMES_PERCENTAGE))
    for i in range(int(total_frames * gv.Params.FRAMES_PERCENTAGE), total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        binary_frame = (frame - mean >= gv.Params.ALPHA * (std + 2)).astype(np.uint8) * 255
        binary_frames.append(binary_frame)
    return binary_frames