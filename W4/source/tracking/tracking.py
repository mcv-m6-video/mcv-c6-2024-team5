import numpy as np
from .sort import Sort


class Tracking:
    def __init__(self):
        self.tracker = Sort(iou_threshold=0.3)

    def __call__(self, preds):
        np_pred = np.array(preds)
        if len(np_pred) == 0:
            np_pred = np.empty((0, 5))
        pred_with_ids = self.tracker.update(np_pred)
        # Transform them to list of lists
        pred_with_ids = pred_with_ids.astype(int).tolist()
        # Insert the confidence back in the 5th position of the list (index 4)
        for pred_with_id, p in zip(pred_with_ids, preds):
            aux = pred_with_id[4]
            pred_with_id[4] = p[4]
            pred_with_id.append(aux)
        return pred_with_ids
