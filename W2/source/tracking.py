import numpy as np
from source.metrics import calculate_iou
import cv2
import source.global_variables as gv
from source.visualization import put_text_top_left, add_rectangles_to_frame_with_id
from sort.sort import Sort
from filterpy.kalman import KalmanFilter
import torch
import pickle


def overlap_tracking(preds):
    # perform tracking of the frames by checking the overlap of the bounding boxes present in preds between frame n and frame n-1. If the overlap is greater than 0.5, the bounding boxes are considered to be the same object.
    # the function assign an id to each bounding box and returns the list of bounding boxes with their respective id

    # Initialize variables
    ids = 1
    new_preds = []

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
        # Check if there are repeated ids
        # if len(pred_ids) != len(set(pred_ids)):
        #     # Get repeated ids
        #     repeated_ids = [id for id in pred_ids if pred_ids.count(id) > 1]
        #     for repeated_id in repeated_ids:
        #         found = False
        #         for id in pred_ids:
        #             if id == repeated_id and not found:
        #                 found = True
        #             elif id == repeated_id and found:
        #                 pred_ids[pred_ids.index(id)] = ids
        #                 ids += 1

        for pred, pred_id in zip(pred, pred_ids):
            new_preds.append([*pred, pred_id])
        prev_ids = np.copy(pred_ids)
    return new_preds


def show_tracking(cap, ids_tracking, total_frames, gt, preds, aps, map):
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        ids = ids_tracking[i]
        # combine binary_frame as an overlay of frame with the binarization in pink color
        overlay = add_rectangles_to_frame_with_id(frame, gt[i], (0, 0, 255))
        overlay = add_rectangles_to_frame_with_id(overlay, preds[i], (0, 255, 0), ids)
        put_text_top_left(overlay, f"AP: {aps[i]:.5f}, Frame: {i}. mAP of full video: {map:.5f}")
        cv2.imshow('overlay', overlay)
        cv2.waitKey(1)


def show_tracking_kalman(cap, binary_frames, trackers, ids_list, total_frames, gt, preds, aps, map):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * gv.Params.FRAMES_PERCENTAGE))
    for i in range(int(total_frames * gv.Params.FRAMES_PERCENTAGE), total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        ids = ids_list[i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)]
        
        binary_frame = binary_frames[i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)]

        # combine binary_frame as an overlay of frame with the binarization in pink color
        binary_frame_color = np.zeros(frame.shape, dtype=np.uint8)
        
        # fill the binary_frame_color with the binary_frame items that were not zeros as pink
        binary_frame_color[binary_frame != 0] = [255, 0, 255]
        overlay = cv2.addWeighted(frame, 0.7, binary_frame_color, 0.3, 0)

        overlay = add_rectangles_to_frame_with_id(overlay, gt[i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)], (0, 0, 255))
        overlay = add_rectangles_to_frame_with_id(overlay, preds[i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)], (0, 255, 0), ids)
        overlay = add_rectangles_to_frame_with_id(overlay, trackers[i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)], (255, 0, 0), ids)
        put_text_top_left(overlay, f"AP: {aps[i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)]:.5f}, Frame: {i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)}. mAP of full video: {map:.5f}")
        cv2.imshow('overlay', overlay)
        cv2.waitKey(0)


def read_bounding_boxes_from_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        preds = []
        for bounding_boxes_list in data:
            frame_preds = []
            for box in bounding_boxes_list:
                frame_preds.append((*box, 1))  
            preds.append(frame_preds)
        return preds


def tracking_kalman_sort(preds):
    # Track objects using Kalman filter and display with binary frames
    tracker = Sort(iou_threshold=0.5)  # Create Sort tracker
    new_preds = []
    for pred in preds:
        np_pred = np.array(pred)
        pred_with_ids = tracker.update(np_pred)
        # Transform them to list of lists
        pred_with_ids = pred_with_ids.astype(int).tolist()
        new_preds.append(pred_with_ids)
    return new_preds

