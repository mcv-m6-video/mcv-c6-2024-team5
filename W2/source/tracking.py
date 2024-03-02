import numpy as np
from source.metrics import calculate_iou
import cv2
import source.global_variables as gv
from source.visualization import put_text_top_left

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
                    if max_iou >= 0.8:
                        pred_ids.append(prev_ids[max_iou_idx])
                    else:
                        pred_ids.append(ids)
                        ids += 1
                        
        prev_ids = np.copy(pred_ids)
        list_of_ids.append(pred_ids)

    return list_of_ids

def add_rectangles_to_frame_with_id(frame, boxes, color, ids):
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
        overlay = add_rectangles_to_frame_with_id(overlay, gt[i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)], (0, 0, 255), ids)
        overlay = add_rectangles_to_frame_with_id(overlay, preds[i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)], (0, 255, 0), ids)
        put_text_top_left(overlay, f"AP: {aps[i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)]:.5f}, Frame: {i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)}. mAP of full video: {map:.5f}")
        cv2.imshow('overlay', overlay)
        cv2.waitKey(0)
    
