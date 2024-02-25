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
        #? If condition is true, then set the pixel to 255, so it is white -> Foreground
        binary_frame = (abs(frame - mean) >= gv.Params.ALPHA * (std + 2)).astype(np.uint8) * 255
        if gv.Params.ADAPTIVE_MODELLING:
            # Update mean and std only for the pixels classified as Background (0)
            aux_mean = (1 - gv.Params.RHO) * mean + gv.Params.RHO * frame
            aux_std = np.sqrt((1 - gv.Params.RHO) * std ** 2 + gv.Params.RHO * (frame - mean) ** 2)
            mean = np.where(binary_frame == 0, aux_mean, mean)
            std = np.where(binary_frame == 0, aux_std, std)

        binary_frames.append(binary_frame)
    return binary_frames