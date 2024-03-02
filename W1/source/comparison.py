import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import source.global_variables as gv
from source.data_io import gt_bbox, gt_bboxes_comparison, calculate_mAP_comparison
from source.visualization import show_binary_frames, add_rectangles_to_frame, put_text_top_left
from source.metrics import compute_ap
from source.data_io import load_video, load_frame_dict
from scripts.graphics import plot_mAP_comparison

# Import and initialize Params
gv.init()

# Define paths
VIDEO_FILE = './data/AICity_data/train/S03/c010/vdo.avi'
ANNOTATIONS_FILE = './frame_dict.json'

# Load annotations from JSON file
frame_dict = load_frame_dict()

def Background_subtraction(background_subtractor, cap, frame_dict, total_frames, percentage):
    # Calculate the target number of frames
    target_frames = int(total_frames * percentage)  
    frame_counter = 0

    backSub = None

    if background_subtractor == 'MOG':
        backSub = cv2.bgsegm.createBackgroundSubtractorMOG()
    elif background_subtractor == 'MOG2':
        backSub = cv2.createBackgroundSubtractorMOG2(history=200, detectShadows=True)
    elif background_subtractor == 'GMG':
        backSub = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=120, decisionThreshold=0.9)
    elif background_subtractor == 'KNN':
        backSub = cv2.createBackgroundSubtractorKNN(dist2Threshold=400.0, detectShadows=True)
    elif background_subtractor == 'CNT':
        backSub = cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=15, useHistory=True, maxPixelStability=15*60, isParallel=True)
    elif background_subtractor == 'GSOC':
        backSub = cv2.bgsegm.createBackgroundSubtractorGSOC(mc=5)

    # For mAP calculation
    annotations_for_calculation = []  
    all_predictions = []  

    # Get the ground truth bounding boxes for frames based on percentage
    gt_annotations = gt_bboxes_comparison(frame_dict, total_frames, percentage)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Current frame index
        frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  

        # Check if frame index is within the range of ground truth annotations
        if frame_index - 1 >= len(gt_annotations):
            print("Frame index out of range. Exiting loop.")
            break

        # Get annotations for current frame
        gt_boxes = gt_annotations[frame_index - 1]

        # Apply the background subtractor to get the foreground mask
        fgMask = backSub.apply(frame)

        # Apply binary thresholding
        threshold_value = 128
        _, binary_img = cv2.threshold(fgMask, threshold_value, 255, cv2.THRESH_BINARY)

        # Find contours of objects in the binary image
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bboxes_pred = []
        for contour in contours:
            # Calculate bounding box coordinates
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter out small detections
            min_area_threshold = 600  
            if cv2.contourArea(contour) < min_area_threshold:
                continue

            # Filter out vertical boxes (bikes)
            aspect_ratio = w / h
            if aspect_ratio < 1.0:  
                continue

            # Filter out big horizontal boxes 
            aspect_ratio = w / h
            if aspect_ratio > 2.5:  
                continue

            # Add annotations for mAP calculation
            annotations_for_calculation.append((x, y, x + w, y + h))

            # Add bounding box for predictions
            bboxes_pred.append((x, y, x + w, y + h))

        all_predictions.append(bboxes_pred)

        # Draw bounding boxes on binary image
        binary_with_boxes = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        for xtl, ytl, xbr, ybr in bboxes_pred:
            cv2.rectangle(binary_with_boxes, (xtl, ytl), (xbr, ybr), (0, 255, 0), 2)  # Green for predicted
        for xtl, ytl, xbr, ybr in gt_boxes:
            cv2.rectangle(binary_with_boxes, (xtl, ytl), (xbr, ybr), (0, 0, 255), 2)  # Red for ground truth

        # Draw bounding boxes on original frame
        frame_with_boxes = frame.copy()
        for xtl, ytl, xbr, ybr in bboxes_pred:
            cv2.rectangle(frame_with_boxes, (xtl, ytl), (xbr, ybr), (0, 255, 0), 2)  # Green for predicted
        for xtl, ytl, xbr, ybr in gt_boxes:
            cv2.rectangle(frame_with_boxes, (xtl, ytl), (xbr, ybr), (0, 0, 255), 2)  # Red for ground truth

        cv2.imshow('Original Image with Bounding Boxes', frame_with_boxes)
        cv2.imshow('Binary Image with Bounding Boxes', binary_with_boxes)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Increment frame counter
        frame_counter += 1
        if frame_counter >= target_frames:
            break

    # Calculate mAP
    aps, mAP = calculate_mAP_comparison(gt_annotations, all_predictions)
    print("Mean Average Precision (mAP):", mAP)

    # Release the VideoCapture and destroy windows
    cap.release()
    cv2.destroyAllWindows()

    # Return the calculated mAP
    return mAP  


# List of background subtraction methods to compare
background_subtractors = ['MOG', 'MOG2', 'GMG', 'KNN', 'CNT', 'GSOC']
# Store mAP results for each method
mAP_results = []  

# Run background subtraction for each method with the chosen percentage
for method in background_subtractors:
    print('Running method: ', method)
    # Create a new VideoCapture for each method
    cap = load_video()  
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    percentage = gv.Params.FRAMES_PERCENTAGE  
    mAP = Background_subtraction(method, cap, frame_dict, total_frames, percentage)
    mAP_results.append((method, percentage, mAP))

# Plot the mAP results for comparison
plot_mAP_comparison(mAP_results, percentage)