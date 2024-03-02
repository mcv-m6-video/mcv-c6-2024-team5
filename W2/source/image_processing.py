import cv2
import numpy as np
import source.global_variables as gv

## PROCESSING FUNCTIONS

def get_relevant_color(frame):
    if gv.Params.COLOR:
        if gv.Params.COLOR_SPACE == "rgb":
            return frame
        elif gv.Params.COLOR_SPACE == "hsv":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        elif gv.Params.COLOR_SPACE == "yuv":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        elif gv.Params.COLOR_SPACE == "lab":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        elif gv.Params.COLOR_SPACE == "ycrcb":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    else:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def process_frames(cap, total_frames):
    frames = []
    print(f"Processing {int(total_frames * gv.Params.FRAMES_PERCENTAGE)} frames for background modelling ({int(gv.Params.FRAMES_PERCENTAGE * 100)}% of the video)")
    for i in range(int(total_frames * gv.Params.FRAMES_PERCENTAGE)):
        ret, frame = cap.read()
        if not ret:
            break
        frame = get_relevant_color(frame)
        
        # Log each 50 frames
        if i % 50 == 0:
            print(f"Processing frame {i}/{int(total_frames * gv.Params.FRAMES_PERCENTAGE)} for background modelling")

        frames.append(frame)
        
    return np.array(frames)

def compute_mean_std(frames):   
    # Incremental computation for color frames
    mean = np.zeros_like(frames[0], dtype=np.float64)
    sq_mean = np.zeros_like(frames[0], dtype=np.float64)
    for frame in frames:
        frame = frame.astype(np.float64)
        mean += frame
        sq_mean += np.square(frame)
    mean /= len(frames)
    std = np.sqrt((sq_mean / len(frames)) - np.square(mean))
    
    # ! OLD and slow method (also RAM consuming mostly for color frames)
    # mean = np.mean(frames, axis=0)
    # std = np.std(frames, axis=0)

    return mean.astype(np.float32), std.astype(np.float32)

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
        frame = get_relevant_color(frame)

        # Compute absolute difference and threshold
        if gv.Params.COLOR:
            # Process each color channel
            binary_frame = np.zeros_like(frame, dtype=np.uint8)
            for c in range(frame.shape[2]):  # Assuming frame is (H, W, C)
                channel_diff = abs(frame[:,:,c] - mean[:,:,c])
                binary_channel = (channel_diff >= gv.Params.ALPHA * (std[:,:,c] + 2)).astype(np.uint8) * 255
                binary_frame[:,:,c] = binary_channel
            # Combine binary masks from all channels
            binary_frame = np.max(binary_frame, axis=2).astype(np.uint8)
        else:
            #? If condition is true, then set the pixel to 255, so it is white -> Foreground
            binary_frame = (abs(frame - mean) >= gv.Params.ALPHA * (std + 2)).astype(np.uint8) * 255

        if gv.Params.ADAPTIVE_MODELLING:
            # Update mean and std only for the pixels classified as Background (0)
            if gv.Params.COLOR:
                for c in range(frame.shape[2]):
                    aux_mean = (1 - gv.Params.RHO) * mean[:,:,c] + gv.Params.RHO * frame[:,:,c]
                    aux_std = np.sqrt((1 - gv.Params.RHO) * std[:,:,c] ** 2 + gv.Params.RHO * (frame[:,:,c] - mean[:,:,c]) ** 2)
                    update_mask = binary_frame == 0
                    mean[:,:,c] = np.where(update_mask, aux_mean, mean[:,:,c])
                    std[:,:,c] = np.where(update_mask, aux_std, std[:,:,c])
            else:
                aux_mean = (1 - gv.Params.RHO) * mean + gv.Params.RHO * frame
                aux_std = np.sqrt((1 - gv.Params.RHO) * std ** 2 + gv.Params.RHO * (frame - mean) ** 2)
                mean = np.where(binary_frame == 0, aux_mean, mean)
                std = np.where(binary_frame == 0, aux_std, std)

        binary_frame = post_processing(binary_frame)
        binary_frames.append(binary_frame)
    return binary_frames

def post_processing(binary_frame):
    # circular kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_CLOSE, kernel)
    binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, kernel)
    # Gaussian blur to smooth the edges, remove noise and fill some gaps
    binary_frame = cv2.GaussianBlur(binary_frame, (5, 5), 0)
    binary_frame = cv2.threshold(binary_frame, 50, 50, cv2.THRESH_BINARY)[1]
    # Close with a vertical kernel to join the low part of the cars with the high part
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((53, 3), np.uint8)
    binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((3, 53), np.uint8)
    binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_CLOSE, kernel)
    return binary_frame

def predict_bboxes(binary_frames):
    bounding_boxes = []
    for binary_frame in binary_frames:
        bounding_boxes.append(find_connected_components(binary_frame))
    return bounding_boxes

def find_connected_components(binary_frame):
    """Finds connected components in a binary frame and returns a list of bounding boxes"""
    conn, labels, values, centroids = cv2.connectedComponentsWithStats(binary_frame)
    # Filter out small connected components
    height, width = binary_frame.shape
    bounding_boxes = []
    for label in range(1, conn):  # Skip background label (0)
        x, y, w, h, area = values[label]
        # Hand-picked thresholds (can be adjusted)
        if area > 750 and area < 1000000 and w > h and w/h <3:
            bounding_boxes.append((x, y, x + w, y + h))
        elif area > 250 and area < 1000000 and  y <height/7 and w > h and h/w <3:
            bounding_boxes.append((x, y, x + w, y + h))
    return bounding_boxes

def background_subtraction_method(background_subtractor):
    if background_subtractor == 'MOG':
        backSub = cv2.bgsegm.createBackgroundSubtractorMOG()
    elif background_subtractor == 'MOG2':
        backSub = cv2.createBackgroundSubtractorMOG2(history=200, detectShadows=True)
    elif background_subtractor == 'GMG':
        backSub = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=120, decisionThreshold=0.9)
    elif background_subtractor == 'KNN':
        backSub = cv2.createBackgroundSubtractorKNN(dist2Threshold=400.0, detectShadows=True)
    elif background_subtractor == 'CNT':
        backSub = cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=15, useHistory=True,
                                                           maxPixelStability=15 * 60, isParallel=True)
    elif background_subtractor == 'GSOC':
        backSub = cv2.bgsegm.createBackgroundSubtractorGSOC(mc=5)
    else:
        raise ValueError("Invalid background subtraction method")
    return backSub

def soa_binary_frames(cap, total_frames, method):
    binary_frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * gv.Params.FRAMES_PERCENTAGE))
    backSub = background_subtraction_method(method)
    for i in range(int(total_frames * gv.Params.FRAMES_PERCENTAGE), total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = get_relevant_color(frame)
        binary_frame = backSub.apply(frame)

        threshold_value = 128
        _, binary_frame = cv2.threshold(binary_frame, threshold_value, 255, cv2.THRESH_BINARY)
        # binary_frame = post_processing(binary_frame)
        # _, binary_frame = cv2.threshold(binary_frame, 1, 255, cv2.THRESH_BINARY)
        binary_frames.append(binary_frame)
    return binary_frames
