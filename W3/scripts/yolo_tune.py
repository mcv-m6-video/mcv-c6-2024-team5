from ultralytics import YOLO

model = YOLO('yolov8n.pt')

path_to_dataset = '/W2/scripts/dataset.yaml'

# Tune the model
model.tune(iterations=25, data=path_to_dataset, imgsz=736, epochs=30, plots=True, save=True, val=True)
