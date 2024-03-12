import os
import cv2
import numpy as np
import source.global_variables as gv
from source.metrics import calculate_iou
from sort.sort import Sort


def get_of_vector(bbox, of_image):
    """
    Calculate the optical flow vector for the bbox
    :param bbox: prediction bbox
    :param of_image: image with the optical flow
    :return: [u, v] vector
    """
    # Get the optical flow for the box
    of_box = of_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    # Calculate the mean value for each channel in the optical flow box image
    _, v, u = cv2.split(of_box)
    mean_v = np.mean(v) - 128
    mean_u = np.mean(u) - 128
    return np.array([mean_u, mean_v])


def calculate_iou_with_of(i, box, previous_preds):
    """
    Calculate a score based on the IOU between the boxes and the optical flow vector
    :param i: Frame number
    :param box: Bounding box of the current frame
    :param box_prev: Bounding box of the previous frame
    :return:
    """
    # Get the optical flow image: the name of the image is "of_to_XXXXXX.png", where XXXXXX is the frame number
    of_image = cv2.imread(f"{gv.PATH_TO_OF}/of_to_{str(i).zfill(6)}.png")
    # Resize the optical flow image to the same size as the predictions
    of_image = cv2.resize(of_image, (1920, 1080))
    # Get the optical flow vector for the box
    of_vector = get_of_vector(box, of_image)
    ious = [calculate_iou(box, box_prev) for box_prev in previous_preds]
    if abs(of_vector[0]) <= 1 and abs(of_vector[1]) <= 1:
        # If the optical flow vector is too small, we resort to the normal IOU approach
        dot_products = [0 for _ in previous_preds]
        return ious, dot_products
    else:
        # If the optical flow vector is not too small, we calculate the dot product between the optical flow vector and
        # the difference between the centers of the boxes. This, added to the IOU, gives us a score that we
        # can use to compare the boxes.
        dot_products = []
        for box_prev in previous_preds:
            center_prev = [(box_prev[0] + box_prev[2]) / 2, (box_prev[1] + box_prev[3]) / 2]
            center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
            center_diff = np.array(center) - np.array(center_prev)
            # Normalize both vectors
            center_diff = center_diff / np.linalg.norm(center_diff)
            of_vector = of_vector / np.linalg.norm(of_vector)
            dot_product = np.dot(of_vector, center_diff)
            dot_products.append(dot_product)
        return ious, dot_products


def overlap_tracking(preds, threshold=0.25, use_of=False):
    # perform tracking of the frames by checking the overlap of the bounding boxes present in preds between frame n and frame n-1. If the overlap is greater than 0.5, the bounding boxes are considered to be the same object.
    # the function assign an id to each bounding box and returns the list of bounding boxes with their respective id

    # Initialize variables
    ids = 1
    new_preds = []

    # Iterate over the preds
    for i, pred in enumerate(preds):
        pred_ids = []
        pred_ids_used = []
        if i == 0:
            for j, box in enumerate(pred):
                pred_ids.append(ids)
                ids += 1
        else:
            for j, box in enumerate(pred):

                if not use_of:
                    scores = [calculate_iou(box, box_prev) for box_prev in preds[i-1]]
                else:
                    ious, dot_products = calculate_iou_with_of(i, box, preds[i - 1])
                    scores = [iou + dot_product for iou, dot_product in zip(ious, dot_products)]
                if len(scores) == 0:
                    pred_ids.append(ids)
                    ids += 1
                else:
                    max_score = max(scores)
                    max_score_idx = scores.index(max_score)
                    if use_of:
                        max_score = ious[max_score_idx]
                    if max_score >= threshold and prev_ids[max_score_idx] not in pred_ids_used:
                        pred_ids.append(prev_ids[max_score_idx])
                        pred_ids_used.append(prev_ids[max_score_idx])
                    else:
                        pred_ids.append(ids)
                        ids += 1

        changed_preds = []
        for pred, pred_id in zip(pred, pred_ids):
            changed_preds.append([*pred, pred_id])
        prev_ids = np.copy(pred_ids)
        new_preds.append(changed_preds)
        if i % 100 == 0:
            print(f"Frame {i} done")
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


def overlap_plus_of_tracking(preds):
    """
    Use the optical flow to improve the tracking
    :param preds: Predictions made by the model
    :return: List of predictions with the ids of the objects
    """
    of_images = os.listdir(gv.PATH_TO_OF)
    of_images.sort()

    ids = 1
    new_preds = []
    pred_ids = []

    for j, bbox in enumerate(preds[0]):
        pred_ids.append(ids)
        ids += 1

    changed_preds = []
    for pred, pred_id in zip(preds[0], pred_ids):
        changed_preds.append([*pred, pred_id])
    new_preds.append(changed_preds)

    for i, pred in enumerate(preds[1:]):
        if i < len(of_images):
            # Load the optical flow image
            of_image = cv2.imread(f"{gv.PATH_TO_OF}/{of_images[i + 1]}")
            # Resize the optical flow image to the same size as the predictions
            of_image = cv2.resize(of_image, (1920, 1080))
            # Apply the optical flow to the predictions
            for j, box in enumerate(pred):
                # Get the optical flow for the box
                of_box = of_image[box[1]:box[3], box[0]:box[2]]
                # cv2.imshow(f"BBox {j} OF", of_box)
                # cv2.waitKey(0)
                # Calculate the mean value for each channel in the optical flow box image
                _, v, u = cv2.split(of_box)
                mean_v = np.mean(v) - 128
                mean_u = np.mean(u) - 128
                a = 1

                # Calculate the optical flow vector

