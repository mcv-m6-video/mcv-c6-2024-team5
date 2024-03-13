from ultralytics import YOLO


path_to_dataset = 'dataset.yaml'
model = YOLO('yolov8n.pt')

# Train the model
model.train(data=path_to_dataset, imgsz=640, name="train_S04", epochs=15)
