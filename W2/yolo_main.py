import cv2
from ultralytics import YOLO

from source.visualization import display_frame_with_overlay
from source.data_io import load_frame_dict, gt_bbox, yolo_bboxes, save_gif_from_overlayed_frames, calculate_mAP
from source.metrics import compute_ap

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
display = False
overlayed_frames = []
aps = []
gts_boxes = []
preds_boxes = []
frames = []
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
    frames.append(frame)
    # Get bboxes from the result
    pred_boxes = yolo_bboxes(result.boxes)
    preds_boxes.append(pred_boxes)
    # Get the GT annotations for the frame
    gt_boxes = gt_bbox(annotations, frame_number)
    gts_boxes.append(gt_boxes)
    # Calculate AP for the frame
    ap = compute_ap(gt_boxes, pred_boxes)
    aps.append(ap)

    frame_number += 1

    # Stop the program if Q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        break
    # ? Temporal: Break at 10 frames
    if frame_number > 50:
        break

# Compute the mAP for the video
map = calculate_mAP(gts_boxes, preds_boxes, aps)
print("mAP of the video:", map)
if visualize:
    for frame, gt_boxes, pred_boxes, ap in zip(frames, gts_boxes, preds_boxes, aps):
        frame_overlay = display_frame_with_overlay(frame, gt_boxes, pred_boxes, ap, map, display)
        overlayed_frames.append(frame_overlay)
# Build and save a gif with the overlayed frames
save_gif_from_overlayed_frames(overlayed_frames, 4)