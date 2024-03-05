from ultralytics import YOLO

model = YOLO('yolov8n.pt')

path_to_dataset = '/home/franmuline/Master_Workspace/C6/mcv-c6-2024-team5/W2/yolo/dataset.yaml'

# Tune the model
model.tune(iterations=25, data=path_to_dataset, imgsz=736, epochs=30, plots=True, save=True, val=True)
