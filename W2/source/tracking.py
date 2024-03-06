import numpy as np
from source.metrics import calculate_iou
from sort.sort import Sort


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
        changed_preds = []
        for pred, pred_id in zip(pred, pred_ids):
            changed_preds.append([*pred, pred_id])
        prev_ids = np.copy(pred_ids)
        new_preds.append(changed_preds)
    return new_preds





def tracking_kalman_sort(preds):
    # Track objects using Kalman filter and display with binary frames
    tracker = Sort(iou_threshold=0.5)  # Create Sort tracker
    new_preds = []
    for pred in preds:
        np_pred = np.array(pred)
        pred_with_ids = tracker.update(np_pred)
        # Transform them to list of lists
        pred_with_ids = pred_with_ids.astype(int).tolist()
        # Insert the confidence back in the 5th position of the list (index 4)
        for pred_with_id, p in zip(pred_with_ids, pred):
            aux = pred_with_id[4]
            pred_with_id[4] = p[4]
            pred_with_id.append(aux)
        new_preds.append(pred_with_ids)
    return new_preds

