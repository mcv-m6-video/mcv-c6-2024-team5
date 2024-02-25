import cv2
import numpy as np
import json

## LOADING AND SAVING FUNCTIONS

def load_video(path_to_video):
    cap = cv2.VideoCapture(path_to_video)
    if not cap.isOpened():
        raise ValueError("Error opening video file")
    return cap

def load_frame_dict(path_to_output):
    with open(f"{path_to_output}frame_dict.json") as file:
        frame_dict = json.load(file)
    return frame_dict

def load_mean_std(path_to_tmp):
    mean = np.load(f"{path_to_tmp}mean.npy")
    std = np.load(f"{path_to_tmp}std.npy")
    return mean, std

## SAVING FUNCTIONS

def save_mean_std(mean, std, path_to_tmp):
    np.save(f"{path_to_tmp}mean.npy", mean)
    np.save(f"{path_to_tmp}std.npy", std)

def save_visualizations(mean_to_viz, std_to_viz, path_to_output):
    cv2.imwrite(f"{path_to_output}mean.png", mean_to_viz)
    cv2.imwrite(f"{path_to_output}std.png", std_to_viz)

## PROCESSING FUNCTIONS

def process_frames(cap, total_frames, frames_percentage):
    frames = []
    for i in range(int(total_frames * frames_percentage)):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
    return np.array(frames)

def compute_mean_std(frames):
    mean = np.mean(frames, axis=0)
    std = np.std(frames, axis=0)
    return mean, std

def truncate_values(mean, std):
    mean_to_viz = np.clip(mean, 0, 255).astype(np.uint8)
    std_to_viz = np.clip(std, 0, 255).astype(np.uint8)
    return mean_to_viz, std_to_viz

def generate_binary_frames(cap, total_frames, frames_percentage, mean, std, alpha):
    binary_frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * frames_percentage))
    for i in range(int(total_frames * frames_percentage), total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        binary_frame = (frame - mean >= alpha * (std + 2)).astype(np.uint8) * 255
        binary_frames.append(binary_frame)
    return binary_frames

## VISUALIZATION FUNCTIONS

def show_binary_frames(binary_frames, frame_dict, total_frames, frames_percentage):
    for i in range(int(total_frames * frames_percentage), total_frames):
        binary_frame = binary_frames[i - int(total_frames * frames_percentage)]
        binary_frame_to_viz = add_rectangles_to_frame(binary_frame, frame_dict, i)
        cv2.imshow('binary_frame', binary_frame_to_viz)
        cv2.waitKey(0)

def add_rectangles_to_frame(frame, frame_dict, frame_number):
    for car_id, car in frame_dict[str(frame_number)].items():
        if car['is_parked']:
            continue
        x1, y1, x2, y2 = int(car['xtl']), int(car['ytl']), int(car['xbr']), int(car['ybr'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return frame

## MAIN AND CONSTANTS

PATH_TO_DATA = './data/AICity_data/'
PATH_TO_VIDEO = f"{PATH_TO_DATA}train/S03/c010/vdo.avi"
PATH_TO_TMP = "./tmp/"
PATH_TO_OUTPUT = "./output/"

MEAN_STD_COMPUTED = True
FRAMES_PERCENTAGE = 0.25
ALPHA = 10

def main():
    cap = load_video(PATH_TO_VIDEO)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_dict = load_frame_dict(PATH_TO_OUTPUT)
    
    if not MEAN_STD_COMPUTED:
        frames = process_frames(cap, total_frames, FRAMES_PERCENTAGE)
        mean, std = compute_mean_std(frames)
        save_mean_std(mean, std, PATH_TO_TMP)
    else:
        mean, std = load_mean_std(PATH_TO_TMP)
    
    mean_to_viz, std_to_viz = truncate_values(mean, std)
    save_visualizations(mean_to_viz, std_to_viz, PATH_TO_OUTPUT)
    
    binary_frames = generate_binary_frames(cap, total_frames, FRAMES_PERCENTAGE, mean, std, ALPHA)
    show_binary_frames(binary_frames, frame_dict, total_frames, FRAMES_PERCENTAGE)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
