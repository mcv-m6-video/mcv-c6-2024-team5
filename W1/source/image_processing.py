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

        binary_frame = post_processing(binary_frame)
        binary_frames.append(binary_frame)
    return binary_frames

def post_processing(binary_frame):
    # cv2.namedWindow('Before', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('After', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Before', 540, 960)
    # cv2.resizeWindow('After', 540, 960)
    # cv2.imshow('Before', binary_frame)
    # Erotion to remove noise
    kernel = np.ones((5, 5), np.uint8)
    binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_ERODE, kernel)
    # Gaussian blur to smooth the edges, remove noise and fill some gaps
    binary_frame = cv2.GaussianBlur(binary_frame, (19, 19), 0)
    binary_frame = cv2.threshold(binary_frame, 100, 255, cv2.THRESH_BINARY)[1]
    # Close with a vertical kernel to join the low part of the cars with the high part
    kernel = np.ones((53, 3), np.uint8)
    binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_CLOSE, kernel)
    # bboxes = find_connected_components(binary_frame)
    # for bbox in bboxes:
    #     cv2.rectangle(binary_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 2)
    # cv2.imshow('After', binary_frame)
    # cv2.waitKey(1)
    return binary_frame

def find_connected_components(binary_frame):
    """Finds connected components in a binary frame and returns a list of bounding boxes"""
    conn, labels, values, centroids = cv2.connectedComponentsWithStats(binary_frame)
    # Filter out small connected components
    bounding_boxes = []
    for label in range(1, conn):  # Skip background label (0)
        x, y, w, h, area = values[label]
        # Hand-picked thresholds
        if area > 500 and area < 100000 and w > h:
            bounding_boxes.append((x, y, x + w, y + h))
    return bounding_boxes
