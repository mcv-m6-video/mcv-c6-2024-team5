import numpy as np


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
