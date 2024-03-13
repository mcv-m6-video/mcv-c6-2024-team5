import cv2
import os

PATH_TO_DATA = "aic19-track1-mtmc-train/train/"
PATH_TO_YOLO = "yolo_week_3/"
sequence_cameras = {
    "S01": ["c001", "c002", "c003", "c004", "c005"],
    "S03": ["c010", "c011", "c012", "c013", "c014", "c015"],
    # Sequence 4 from 16 to 28
    "S04": ["c016", "c017", "c018", "c019", "c020", "c021", "c022", "c023", "c024", "c025", "c026", "c027", "c028"]
}


def create_yolo_directory_structure(sequence):
    sequence_path = os.path.join(PATH_TO_DATA, sequence)
    cameras = sequence_cameras[sequence]
    # Create directory for images and labels
    os.makedirs(PATH_TO_YOLO + sequence, exist_ok=True)
    os.makedirs(PATH_TO_YOLO + sequence + "/images", exist_ok=True)
    os.makedirs(PATH_TO_YOLO + sequence + "/labels", exist_ok=True)
    for camera in cameras:
        camera_path = os.path.join(sequence_path, camera)
        cap = cv2.VideoCapture(camera_path + "/vdo.avi")
        gt_file = open(camera_path + "/gt/gt.txt", "r")
        # Read the ground truth file and create a dictionary with the frame number as key and the bounding boxes as value
        gt = {}
        for i, line in enumerate(gt_file):
            line = line.split(",")
            frame = int(line[0])
            if frame not in gt:
                gt[frame] = []
            gt[frame].append([int(line[2]), int(line[3]), int(line[4]), int(line[5])])
        counter = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Save the frame
            if counter in gt:
                cv2.imwrite(PATH_TO_YOLO + sequence + "/images/" + camera + "_" + str(counter) + ".jpg", frame)
                img_sz = frame.shape
                # Save the labels
                label_file = open(PATH_TO_YOLO + sequence + "/labels/" + camera + "_" + str(counter) + ".txt", "w")
                for bbox in gt[counter]:
                    x = (bbox[0] + bbox[2] / 2) / img_sz[1]
                    y = (bbox[1] + bbox[3] / 2) / img_sz[0]
                    w = bbox[2] / img_sz[1]
                    h = bbox[3] / img_sz[0]
                    label_file.write("0 " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n")
                label_file.close()
            counter += 1


if __name__ == "__main__":
    sequences = os.listdir(PATH_TO_DATA)
    for sequence in sequences:
        create_yolo_directory_structure(sequence)
