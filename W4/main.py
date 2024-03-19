import time
import os
import yaml
import argparse
from source import VideoCaptureWrapper, YOLOv8
from source.tracking import Tracking


def main(config):
    path_to_sequence = config['sequence']
    # To get video sources, get all cameras in sequence, sort them and add 'vdo.avi' to each camera path
    video_sources = [os.path.join(path_to_sequence, camera, 'vdo.avi') for camera in sorted(os.listdir(path_to_sequence))]
    cap = VideoCaptureWrapper(video_sources)
    model = YOLOv8(config['od_model'])
    tracking_list = [Tracking() for _ in range(cap.num_captures)]
    for i, frames in enumerate(cap):
        t0 = time.time()
        results = model(frames)
        preds_with_ids = [tracking(result) for tracking, result in zip(tracking_list, results)]
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
