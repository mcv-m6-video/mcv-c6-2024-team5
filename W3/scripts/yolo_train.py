from ultralytics import YOLO
import yaml


path_to_dataset = '/W2/scripts/dataset.yaml'
path_to_hyperparameters = '/home/franmuline/Master_Workspace/C6/mcv-c6-2024-team5/W2/scripts/runs/detect/tune/best_hyperparameters.yaml'

with open(path_to_hyperparameters, 'r') as file:
    hyperparameters = yaml.safe_load(file)

model = YOLO('yolov8n.pt')

# Train the model
model.train(data=path_to_dataset, imgsz=736, name="strategy_c_k3", epochs=30, **hyperparameters)
