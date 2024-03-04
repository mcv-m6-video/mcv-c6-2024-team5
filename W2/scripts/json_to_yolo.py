import json
import cv2
import os
import shutil
import random

PATH_TO_JSON = "../frame_dict.json"
PATH_TO_YOLO_FOLDER = "../yolo"
PATH_TO_TRAIN = "/train"
PATH_TO_VALID = "/valid"
PATH_TO_LABELS = "/labels"
PATH_TO_IMAGES = "/images"
PATH_TO_VIDEO = "../data/AICity_data/train/S03/c010/vdo.avi"
IMAGE_SIZE = (1920, 1080)


def gt_bbox(frame_dict, frame_number):
    bboxes = []
    for car_id, car in frame_dict[str(frame_number)].items():
        # Not skip parked cars in W2
        # if car['is_parked']:
        #     continue
        x1, y1, x2, y2 = int(car['xtl']), int(car['ytl']), int(car['xbr']), int(car['ybr'])
        bboxes.append((x1, y1, x2, y2))
    return bboxes


def yolo_format(bbox):
    x = (bbox[0] + bbox[2]) / 2 / IMAGE_SIZE[0]
    y = (bbox[1] + bbox[3]) / 2 / IMAGE_SIZE[1]
    w = (bbox[2] - bbox[0]) / IMAGE_SIZE[0]
    h = (bbox[3] - bbox[1]) / IMAGE_SIZE[1]
    return x, y, w, h


def check_create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_frame(path, frame_dict, frame, frame_number):
    # Check if the folder exists
    check_create_folder(f"{path}{PATH_TO_IMAGES}")
    # Save the frame
    cv2.imwrite(f"{path}{PATH_TO_IMAGES}/frame_{frame_number}.png", frame)
    # Check if the folder exists
    check_create_folder(f"{path}{PATH_TO_LABELS}")
    # Create the label file
    with open(f"{path}{PATH_TO_LABELS}/frame_{frame_number}.txt", "w") as file:
        # Get coordinates
        gts = gt_bbox(frame_dict, frame_number)
        for gt in gts:
            # Change coordinates to YOLO format
            x, y, w, h = yolo_format(gt)
            file.write(f"0 {x} {y} {w} {h}\n")


def strategy_a(cap, frame_dict, percentage=0.25):
    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Get the number of frames to process
    num_frames = int(total_frames * percentage)
    path_to_yolo = PATH_TO_YOLO_FOLDER + "/strategy_a"

    # Read the video
    for frame_number, frame in enumerate(frame_dict.items()):
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            break
        path_to_save = path_to_yolo + PATH_TO_TRAIN if frame_number < num_frames else path_to_yolo + PATH_TO_VALID
        save_frame(path_to_save, frame_dict, frame, frame_number)

        # Log each 50 frames
        if frame_number % 50 == 0:
            print(f"Processing frame {frame_number}/{total_frames} for background modelling")


def copy_files(path_to_extract, path_to_save):
    # Get the files
    files = os.listdir(path_to_extract + PATH_TO_IMAGES)
    # Copy the files
    for file in files:
        shutil.copy(f"{path_to_extract}{PATH_TO_IMAGES}/{file}", f"{path_to_save}{PATH_TO_IMAGES}/{file}")
        shutil.copy(f"{path_to_extract}{PATH_TO_LABELS}/{file.replace('.png', '.txt')}",
                    f"{path_to_save}{PATH_TO_LABELS}/{file.replace('.png', '.txt')}")


def rearrange_k_files(path_to_yolo, k):
    for i in range(0, k):
        path_to_save = path_to_yolo + f"_k{i}" + PATH_TO_VALID
        # Check if the folder exists
        check_create_folder(f"{path_to_save}{PATH_TO_IMAGES}")
        # Do the same for the labels
        check_create_folder(f"{path_to_save}{PATH_TO_LABELS}")
        for j in range(0, k):
            if i == j:
                continue
            path_to_extract = path_to_yolo + f"_k{j}" + PATH_TO_TRAIN
            # Get the files
            copy_files(path_to_extract, path_to_save)


def strategy_b(cap, frame_dict, k=4):
    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Get the number of frames to process
    num_frames = int(total_frames * 1/k)
    path_to_yolo = PATH_TO_YOLO_FOLDER + "/strategy_b"

    for frame_number, frame in enumerate(frame_dict.items()):
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            break

        current_k = frame_number // num_frames
        path_to_save = path_to_yolo + f"_k{current_k if current_k < k - 1 else k - 1}" + PATH_TO_TRAIN
        save_frame(path_to_save, frame_dict, frame, frame_number)

        # Log each 50 frames
        if frame_number % 50 == 0:
            print(f"Processing frame {frame_number}/{total_frames} for background modelling")

    rearrange_k_files(path_to_yolo, k)


def strategy_c(cap, frame_dict, k=4):
    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Get the number of frames to process
    num_frames = int(total_frames * 1/k)
    path_to_yolo = PATH_TO_YOLO_FOLDER + "/strategy_c"

    shuffled_list = [i for i in range(0, total_frames)]
    random.shuffle(shuffled_list)

    for i, frame_number in enumerate(shuffled_list):
        # Read the specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            break

        current_k = i // num_frames
        path_to_save = path_to_yolo + f"_k{current_k if current_k < k - 1 else k - 1}" + PATH_TO_TRAIN
        save_frame(path_to_save, frame_dict, frame, frame_number)

        # Log each 50 frames
        if i % 50 == 0:
            print(f"Processing frame {i}/{total_frames} for background modelling")

    rearrange_k_files(path_to_yolo, k)


if __name__ == "__main__":
    with open(PATH_TO_JSON, "r") as file:
        frame_dict = json.load(file)
    cap = cv2.VideoCapture(PATH_TO_VIDEO)
    # strategy_a(cap, frame_dict)
    # strategy_b(cap, frame_dict)
    strategy_c(cap, frame_dict)
    cap.release()
    cv2.destroyAllWindows()
    print("Done!")




