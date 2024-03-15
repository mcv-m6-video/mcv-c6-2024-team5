import os
import numpy as np
import cv2

DATA_PATH = 'data/aic19-track1-mtmc-train/train/'
# Sequence, camera and FPS
DATA_TREE = {
    'S01': {
        'c001': 10,
        'c002': 10,
        'c003': 10,
        'c004': 10,
        'c005': 10,
    },
    'S03': {
        'c010': 10,
        'c011': 10,
        'c012': 10,
        'c013': 10,
        'c014': 10,
        'c015': 8,
    },
    'S04': {
        'c016': 10,
        'c017': 10,
        'c018': 10,
        'c019': 10,
        'c020': 10,
        'c021': 10,
        'c022': 10,
        'c023': 10,
        'c024': 10,
        'c025': 10,
        'c026': 10,
        'c027': 10,
        'c028': 10
    }
}

# Load calibration matrix
def load_calibration_matrix(calibration_path):
    calibration_mat = open(calibration_path, 'r').read()
    calibration_mat = calibration_mat.split(';')
    for i in range(len(calibration_mat)):
        calibration_mat[i] = calibration_mat[i].split(' ')
        if '\n' in calibration_mat[i]:
            calibration_mat[i].remove('\n')
        # Transform to float
        calibration_mat[i] = list(map(float, calibration_mat[i]))
    calibration_mat = np.reshape(calibration_mat, (3, 3))
    return calibration_mat

def get_rgb_color(id):
    color = (id * 10 % 255, id * 20 % 255, id * 30 % 255)
    scalar_color = tuple([int(c) for c in color])
    return scalar_color

def visualize_bbox_and_speed(fps, gt, calibration_mat, car_last_positions, i, frame):
    for row in gt[gt[:, 0] == i]:
        n_frame, id, left, top, width, height = row[:6].astype(int)
        color = get_rgb_color(id)
        cv2.rectangle(frame, (left, top), (left + width, top + height), color, 2)
        position = np.array([left + width / 2, top + height / 2, 1])
        # Show the center of the bounding box
        cv2.circle(frame, (int(position[0]), int(position[1])), 5, get_rgb_color(id), -1)
        # Compute the real position
        calibration_inv = np.linalg.inv(calibration_mat)
        real_position = np.dot(calibration_inv, position)
        # If the car has a previous position, compute speed
        if id in car_last_positions:
            last_position = car_last_positions[id]
            displacement = np.linalg.norm(real_position - last_position)
            print(f'Movement from {last_position} to {real_position} is {displacement} units for car {id}')
            movement_per_second = displacement * fps
            cv2.putText(frame, str(movement_per_second), (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        car_last_positions[id] = real_position

sequence = 1 # S01, S03, S04
camera = 1 # c001, c002, c003, c004 ... c028
sequence_str = 'S{:02d}'.format(sequence)
camera_str = 'c{:03d}'.format(camera)
fps = DATA_TREE[sequence_str][camera_str]
camera_path = os.path.join(DATA_PATH, sequence_str, camera_str)

# More paths
video_path = os.path.join(camera_path, 'vdo.avi')
gt_path = os.path.join(camera_path, 'gt/gt.txt')
calibration_path = os.path.join(camera_path, 'calibration.txt')

# GT format: frame, ID, left, top, width, height, 1, -1, -1, -1
gt = np.loadtxt(gt_path, delimiter=',')

# Load calibration matrix
calibration_mat = load_calibration_matrix(calibration_path)

# Compute extrinsic matrix


car_last_positions = {}

START_FRAME = 50
# Play video in real time
cap = cv2.VideoCapture(video_path)
# Set the video to the start frame
cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
i = START_FRAME

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Add the gt bounding boxes and speed
    visualize_bbox_and_speed(get_rgb_color, fps, gt, calibration_mat, car_last_positions, i, frame)

    # Wait for the next frame considering the FPS
    wait_time = int(1000 / fps)
    slow_mode = True
    if slow_mode:
        wait_time = 1000
    # Wait for the next frame
    cv2.waitKey(wait_time)  
    cv2.imshow('frame', frame)
    i += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break