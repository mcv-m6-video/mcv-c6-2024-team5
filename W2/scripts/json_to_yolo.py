import json
import cv2
import os

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


def strategy_a(cap, frame_dict, percentage=0.25):
    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Get the number of frames to process
    num_frames = int(total_frames * percentage)

    # Read the video
    for frame_number, frame in enumerate(frame_dict.items()):
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            break
        path_to_yolo = PATH_TO_YOLO_FOLDER + "/strategy_a"
        path_to_save = path_to_yolo + PATH_TO_TRAIN if frame_number < num_frames else path_to_yolo + PATH_TO_VALID
        # Check if the folder exists
        if not os.path.exists(f"{path_to_save}{PATH_TO_IMAGES}"):
            os.makedirs(f"{path_to_save}{PATH_TO_IMAGES}")
        # Save the frame
        cv2.imwrite(f"{path_to_save}{PATH_TO_IMAGES}/frame_{frame_number}.png", frame)
        # Check if the folder exists
        if not os.path.exists(f"{path_to_save}{PATH_TO_LABELS}"):
            os.makedirs(f"{path_to_save}{PATH_TO_LABELS}")
        # Create the label file
        with open(f"{path_to_save}{PATH_TO_LABELS}/frame_{frame_number}.txt", "w") as file:
            # Get coordinates
            gts = gt_bbox(frame_dict, frame_number)
            for gt in gts:
                # Change coordinates to YOLO format
                x, y, w, h = yolo_format(gt)
                file.write(f"0 {x} {y} {w} {h}\n")

        # Log each 50 frames
        if frame_number % 50 == 0:
            print(f"Processing frame {frame_number}/{total_frames} for background modelling")


def strategy_b(cap, frame_dict, k=4):
    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Get the number of frames to process
    num_frames = int(total_frames * 1/k)
    # TODO: ESTÁ A MEDIAS


if __name__ == "__main__":
    with open(PATH_TO_JSON, "r") as file:
        frame_dict = json.load(file)
    cap = cv2.VideoCapture(PATH_TO_VIDEO)
    strategy_a(cap, frame_dict)
    cap.release()
    cv2.destroyAllWindows()
    print("Done!")




