import torch
import torch.nn as nn
import torchvision.transforms as transforms
from ultralytics import YOLO


class YOLOv8(nn.Module):
    def __init__(self, path_to_model):
        super(YOLOv8, self).__init__()
        self.yolo = YOLO(path_to_model)

    def forward(self, x):
        results = self.yolo(x, verbose=False)
        # Convert the results to a list of lists
        f_results = []
        for result in results:
            f_result = []
            for box in result.boxes.cpu().numpy():
                box_list = box.xyxy.tolist()[0]
                # Append the confidence to the list
                conf = box.conf.tolist()[0]
                box_list.append(conf)
                f_result.append(box_list)
            f_results.append(f_result)
        return f_results

