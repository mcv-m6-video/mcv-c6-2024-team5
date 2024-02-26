import cv2
import json
import numpy as np

# Define paths
VIDEO_FILE = './data/AICity_data/train/S03/c010/vdo.avi'
ANNOTATIONS_FILE = './output/frame_dict.json'

# Create a VideoCapture object
cap = cv2.VideoCapture(VIDEO_FILE)

# Load annotations from JSON file
with open(ANNOTATIONS_FILE) as f:
    annotations = json.load(f)

def get_annotations_for_frame(frame_index):
    # Convert string coordinates to integers
    annotations_int = {int(k): v for k, v in annotations.items()}
    return annotations_int.get(frame_index, [])

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes with the format (x1, y1, x2, y2)
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    w_I = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    h_I = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    I = w_I * h_I

    U = w1 * h1 + w2 * h2 - I

    return I / U


def compute_ap(gt_boxes, pred_boxes):
    """
    Compute the average precision (AP) of a model
    :param gt_boxes:
    :param pred_boxes:
    :return:
    """
        
    if not isinstance(gt_boxes, list):
        gt_boxes = []
    if not isinstance(pred_boxes, list):
        pred_boxes = []

    # Initialize variables
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    gt_matched = np.zeros(len(gt_boxes))

    # Iterate over the predicted boxes
    for i, pred_box in enumerate(pred_boxes):
        ious = [calculate_iou(pred_box, gt_box) for gt_box in gt_boxes]
        if len(ious) == 0:
            fp[i] = 1
            continue
        max_iou = max(ious)
        max_iou_idx = ious.index(max_iou)

        if max_iou >= 0.5 and not gt_matched[max_iou_idx]:
            tp[i] = 1
            gt_matched[max_iou_idx] = 1
        else:
            fp[i] = 1

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / len(gt_boxes)
    precision = tp / (tp + fp)

    # Generate graph with the 11-point interpolated precision-recall curve
    recall_interp = np.linspace(0, 1, 11)
    precision_interp = np.zeros(11)
    for i, r in enumerate(recall_interp):
        array_precision = precision[recall >= r]
        if len(array_precision) == 0:
            precision_interp[i] = 0
        else:
            # precision_interp[i] = max(precision[recall >= r])
            precision_interp[i] = max(array_precision)

    ap = np.mean(precision_interp)
    return ap


def compute_video_ap(gt, pred):
    """
    Compute the average precision (AP) of a model for a video
    :param gt: list of lists of ground truth bounding boxes
    :param pred: list of lists of predicted bounding boxes
    :return: list of average precision values
    """
    aps = []
    for i, (gt_boxes, pred_boxes) in enumerate(zip(gt, pred)):
        aps.append(compute_ap(gt_boxes, pred_boxes))
    return aps


def Background_subtraction(background_subtractor):

    target_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * 0.25)  # 25% of total frames
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


    annotations_for_calculation = []  # For mAP calculation
    all_predictions = []  # For mAP calculation

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # Current frame index

        # Get annotations for current frame
        gt_boxes = get_annotations_for_frame(frame_index)

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
            min_area_threshold = 700  # Adjust as needed
            if cv2.contourArea(contour) < min_area_threshold:
                continue

            # # Filter out big detections
            # max_area_threshold = 5000  # Adjust as needed
            # if cv2.contourArea(contour) > max_area_threshold:
            #     continue
      
            # Filter out vertical boxes (assuming bikes)
            aspect_ratio = w / h
            if aspect_ratio < 1.0:  # If taller than wide, likely a bike
                continue

            # Add annotations for mAP calculation
            annotations_for_calculation.append((x, y, x + w, y + h))

            # Add bounding box for predictions
            bboxes_pred.append((x, y, x + w, y + h))

        all_predictions.extend(bboxes_pred)

        # Draw bounding boxes on binary image
        binary_with_boxes = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2RGB)
        for xtl, ytl, xbr, ybr in bboxes_pred:
            cv2.rectangle(binary_with_boxes, (xtl, ytl), (xbr, ybr), (0, 255, 0), 2)  # Green for predicted

        # Draw bounding boxes on original frame
        frame_with_boxes = frame.copy()
        for xtl, ytl, xbr, ybr in bboxes_pred:
            cv2.rectangle(frame_with_boxes, (xtl, ytl), (xbr, ybr), (0, 255, 0), 2)  # Green for predicted

        cv2.imshow('Original Image with Bounding Boxes', frame_with_boxes)
        cv2.imshow('Binary Image with Bounding Boxes', binary_with_boxes)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Increment frame counter
        frame_counter += 1
        if frame_counter >= target_frames:
            break

    # Compute video AP
    # video_aps = compute_video_ap(annotations_for_calculation, all_predictions)
    # print("Video APs:", video_aps)

    # Calculate mAP
    mAP = compute_ap(annotations_for_calculation, all_predictions)
    print("Mean Average Precision (mAP):", mAP)

    # Release the VideoCapture and destroy windows
    cap.release()
    cv2.destroyAllWindows()

# Run the background subtraction with the chosen method
background_subtractor = 'GSOC'  # Change this to the desired method
Background_subtraction(background_subtractor)
