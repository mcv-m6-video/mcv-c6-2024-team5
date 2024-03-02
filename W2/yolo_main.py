import cv2
import json
from ultralytics import YOLO

# Annotations in frame_dict are in the format:
# ```
# {
# [frame_number]: {
#     [object_id]: {
#            "xtl": 1285.84,
#             "ytl": 363.23,
#             "xbr": 1516.36,
#             "ybr": 546.91,
#             "outside": 0,
#             "occluded": 0,
#             "keyframe": 1,
#             "is_parked": true
#     }, ...
# }, ...
# }
# ```
# Where the frame_number and object_id are numbers and the values are just example values.

# Load the YOLO model
yolo_model = YOLO("yolov8n.pt")

# Path to your CCTV video
source = 'data/AICity_data/train/S03/c010/vdo.avi'

# Annotations for the video
annotations = json.load(open('frame_dict.json'))

# Run inference on the source, we'll keep only cars
results = yolo_model.predict(
    source, 
    stream = True,
    device = 'cuda:0',  # GPU inference
    classes = [2]  # Only cars
)  # generator of Results objects

# Process results generator
for result in results:
    # boxes = result.boxes  # Boxes object for bounding box outputs
    # masks = result.masks  # Masks object for segmentation masks outputs
    # keypoints = result.keypoints  # Keypoints object for pose outputs
    # probs = result.probs  # Probs object for classification outputs
    result.show()  # display to screen
    # result.save(filename='result.jpg')  # save to disk
    

    break  # stop iteration
