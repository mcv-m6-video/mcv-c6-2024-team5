import cv2
from ultralytics import YOLO

from source.visualization import add_rectangles_to_frame
from source.data_io import load_frame_dict, gt_bbox, yolo_bboxes

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

models_list = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
]

# Load the YOLO model
yolo_model = YOLO("yolov8n.pt")

# Path to your CCTV video
source = 'data/AICity_data/train/S03/c010/vdo.avi'

# Annotations for the video
annotations = load_frame_dict()

# Run inference on the source, we'll keep only cars
results = yolo_model.predict(
    source, 
    stream = True,
    device = 'cuda:0',  # GPU inference
    classes = [2]  # Only cars
)  # generator of Results objects

frame_number = 0
visualize = True
# Process results generator
for result in results:
    # boxes = result.boxes  # Boxes object for bounding box outputs
    # masks = result.masks  # Masks object for segmentation masks outputs
    # keypoints = result.keypoints  # Keypoints object for pose outputs
    # probs = result.probs  # Probs object for classification outputs
    # result.show()  # display to screen
    # result.save(filename='result.jpg')  # save to disk
    
    # Get original frame
    frame = result.orig_img

    # Get bboxes from the result
    pred_boxes = yolo_bboxes(result.boxes)
    # Get the GT annotations for the frame
    gt_boxes = gt_bbox(annotations, frame_number)
    
    if visualize:
        # Add the GT and predicted boxes to the frame
        frame = add_rectangles_to_frame(frame, gt_boxes, (0, 255, 0)) # Green for the GT
        frame = add_rectangles_to_frame(frame, pred_boxes, (0, 0, 255)) # Red for the predictions
        # Show the frame
        cv2.imshow('frame', frame)
        cv2.waitKey(0)

    frame_number += 1
    break  # stop iteration
