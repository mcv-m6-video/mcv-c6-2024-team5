import numpy as np


def calculate_iou(boxA, boxB):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes with the format (x1, y1, x2, y2)
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def compute_ap(gt_boxes, pred_boxes):
    """
    Compute the average precision (AP) of a model
    :param gt_boxes:
    :param pred_boxes:
    :return:
    """

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
            precision_interp[i] = max(precision[recall >= r])

    ap = np.mean(precision_interp)
    return ap

def compute_ap_confidences(gt_boxes, pred_boxes, pred_confidences, confidence_threshold=0.5, iou_threshold=0.5):
    """
    Compute the average precision (AP) of a model
    :param gt_boxes:
    :param pred_boxes:
    :param pred_confidences:
    :param confidence_threshold:
    :return:
    """

    # Initialize variables
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    gt_matched = np.zeros(len(gt_boxes))

    # Sort the predicted boxes by confidence
    pred_boxes = [box for _, box in sorted(zip(pred_confidences, pred_boxes), reverse=True)]

    # Iterate over the predicted boxes
    for i, pred_box in enumerate(pred_boxes):
        if pred_confidences[i] < confidence_threshold:
            fp[i] = 1
            continue
        ious = [calculate_iou(pred_box, gt_box) for gt_box in gt_boxes]
        if len(ious) == 0:
            fp[i] = 1
            continue
        max_iou = max(ious)
        max_iou_idx = ious.index(max_iou)

        if max_iou >= iou_threshold and not gt_matched[max_iou_idx]:
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
            precision_interp[i] = max(precision[recall >= r])

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
