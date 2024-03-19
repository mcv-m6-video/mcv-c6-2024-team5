import time
import os
import yaml
import argparse
import torch
from source import VideoCaptureWrapper, YOLOv8, TripletEfficientNet
from source.tracking import Tracking


def main(config):
    path_to_sequence = config['sequence']
    # To get video sources, get all cameras in sequence, sort them and add 'vdo.avi' to each camera path
    video_sources = [os.path.join(path_to_sequence, camera, 'vdo.avi') for camera in sorted(os.listdir(path_to_sequence))]
    cap = VideoCaptureWrapper(video_sources)
    od_model = YOLOv8(config['od_model'])
    fe_model = TripletEfficientNet()
    # Load the weights of the model
    fe_model.load_state_dict(torch.load(config['feature_model']))

    tracking_dict = {id: Tracking(embedding_model=fe_model, distance_threshold=config["feature_d_threshold"]) for id in cap.video_captures.keys()}
    for i, frames in enumerate(cap):
        t0 = time.time()
        results = od_model(list(frames.values()))
        preds_with_ids = [tracking_dict[id](result, frame, i) for result, (id, frame) in zip(results, frames.items())]
        cap.collage(preds_with_ids)
        t1 = time.time()
        print(f"Time to process the batch: {t1 - t0}")


if __name__ == "__main__":
    # Read the arguments with argparse
    parser = argparse.ArgumentParser(description='C6 Team 5 - Week 4')
    parser.add_argument('--config', type=str, default="config/S01.yaml", help='Path to the configuration file')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config_file = yaml.safe_load(file)
    main(config_file)
