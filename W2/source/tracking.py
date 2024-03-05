import numpy as np
from source.metrics import calculate_iou
import cv2
import source.global_variables as gv
from source.visualization import put_text_top_left
from sort.sort import Sort

def overlap_tracking(preds):
    # perform tracking of the frames by checking the overlap of the bounding boxes present in preds between frame n and frame n-1. If the overlap is greater than 0.5, the bounding boxes are considered to be the same object.
    # the function assign an id to each bounding box and returns the list of bounding boxes with their respective id

    # Initialize variables
    ids = 1
    list_of_ids = []

    # Iterate over the preds
    for i, pred in enumerate(preds):
        pred_ids = []
        if i == 0:
            for j, box in enumerate(pred):
                pred_ids.append(ids)
                ids += 1
        else:
            for j, box in enumerate(pred):
                
                ious = [calculate_iou(box, box_prev) for box_prev in preds[i-1]]
                if len(ious) == 0:
                    pred_ids.append(ids)
                    ids += 1
                else:
                    max_iou = max(ious)
                    max_iou_idx = ious.index(max_iou)
                    if max_iou >= 0.6:
                        pred_ids.append(prev_ids[max_iou_idx])
                    else:
                        pred_ids.append(ids)
                        ids += 1
                      
        prev_ids = np.copy(pred_ids)
        list_of_ids.append(pred_ids)

    return list_of_ids

def add_rectangles_to_frame_with_id(frame, boxes, color, ids=None):
    if ids is None:
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    else: 
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            id = ids[i]
            cv2.putText(frame, str(id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame


def show_tracking(cap, binary_frames,ids_tracking, total_frames, gt, preds, aps, map):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * gv.Params.FRAMES_PERCENTAGE))
    for i in range(int(total_frames * gv.Params.FRAMES_PERCENTAGE), total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        ids = ids_tracking[i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)]
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #frame = frames[i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)]
        binary_frame = binary_frames[i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)]

        # combine binary_frame as an overlay of frame with the binarization in pink color
        binary_frame_color = np.zeros(frame.shape, dtype=np.uint8)
        
        # fill the binary_frame_color with the binary_frame items that were not zeros as pink
        binary_frame_color[binary_frame != 0] = [255, 0, 255]
        overlay = cv2.addWeighted(frame, 0.7, binary_frame_color, 0.3, 0)

        overlay = add_rectangles_to_frame_with_id(overlay, gt[i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)], (0, 0, 255))
        overlay = add_rectangles_to_frame_with_id(overlay, preds[i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)], (0, 255, 0), ids)
        put_text_top_left(overlay, f"AP: {aps[i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)]:.5f}, Frame: {i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)}. mAP of full video: {map:.5f}")
        cv2.imshow('overlay', overlay)
        cv2.waitKey(0)


def tracking_kalman(cap, binary_frames, preds, total_frames, frame_dict):
    # Track objects using Kalman filter and display with binary frames

    tracker = Sort()  # Create Sort tracker
    num_trackers = 0  # Initialize tracker count
    trackers = []  # Initialize trackers

    for i in range(int(total_frames * gv.Params.FRAMES_PERCENTAGE), min(total_frames, len(preds))):  
        # Update trackers if there are predictions
        if len(preds[i]) > 0:
            trackers = tracker.update(np.array(preds[i]))  
            num_trackers += len(trackers)  

        print("Processing frame init:", i)  

        # Check if frame number exists in frame_dict
        if i in frame_dict:
            frame_info = frame_dict[i]
            frame = frame_info['frame']
            boxes_gt = frame_info['bbox']
        else:
            continue

        # Draw ground truth bounding boxes (Red Boxes)
        for box in boxes_gt:
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)

        # Display the tracked boxes (Blue Boxes)
        for d in trackers:
            d = d.astype(np.int32)
            cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), (255, 0, 0), 2)  
            cv2.putText(frame, str(d[4]), (d[0], d[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  

        # Combine binary_frame as an overlay of frame with the binarization in pink color
        binary_frame_color = np.zeros(frame.shape, dtype=np.uint8)
        binary_frame_color[binary_frames[i] != 0] = [255, 0, 255]
        overlay = cv2.addWeighted(frame, 0.7, binary_frame_color, 0.3, 0)

        # Add rectangles to overlay with IDs
        ids = []  
        for pred in preds[i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)]:
            ids.append(int(pred[-1]))  
        overlay_with_ids = add_rectangles_to_frame_with_id(overlay, preds[i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)], (0, 255, 0), ids)

        # Add text to overlay
        ap = 0.0 if i - int(total_frames * gv.Params.FRAMES_PERCENTAGE) >= len(aps) else aps[i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)]
        put_text_top_left(overlay_with_ids, f"AP: {ap:.5f}, Frame: {i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)}. mAP of full video: {map:.5f}")

        # Display the overlay
        cv2.imshow('Tracking with a Kalman filter', overlay_with_ids)
        key = cv2.waitKey(0)  

        print("Processing frame end:", i)  

    print("Finished processing all frames")

    return trackers, num_trackers




