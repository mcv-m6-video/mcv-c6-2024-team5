import os
import numpy as np
import cv2

DATA_PATH = 'data/aic19-track1-mtmc-train/train/'

# Sequence, camera, and FPS 
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

# Function to visualize bounding box and calculate speed
def visualize_bbox_and_speed(fps, gt, calibration_inv, car_last_positions, conversion_factor, i, frame, speeds_dict):
    # Draw bounding box for fire hydrant
    cv2.rectangle(frame, (int(xtl), int(ytl)), (int(xbr), int(ybr)), (0, 255, 255), 2)
    for row in gt[gt[:, 0] == i]:
        n_frame, id, left, top, width, height = row[:6].astype(int)
        color = get_rgb_color(id)
        cv2.rectangle(frame, (left, top), (left + width, top + height), color, 2)
        position = np.array([left + width / 2, top + height / 2, 1])
        # Show the center of the bounding box
        cv2.circle(frame, (int(position[0]), int(position[1])), 5, get_rgb_color(id), -1)
        # Compute the real position
        real_position = np.dot(calibration_inv, position)
        # If the car has a previous position, compute speed
        if id in car_last_positions:
            last_position = car_last_positions[id]
            displacement = np.linalg.norm(real_position - last_position)
            # Convert displacement from pixels to meters using conversion factor
            displacement_meters = displacement * conversion_factor
            print(f'Movement from {last_position} to {real_position} is {displacement} units and {displacement_meters} meters for car {id}')
            # Calculate speed in m/s
            speed_mps = displacement_meters * fps 
            # Convert speed to km/h
            speed_kmh = speed_mps * 3.6  # 1 m/s = 3.6 km/h
            print(f'Speed: {speed_kmh} km/h for car {id}')
            # Store speed for this ID
            if id in speeds_dict:
                speeds_dict[id].append(speed_kmh)
            else:
                speeds_dict[id] = [speed_kmh]
            # Display speed on the frame
            cv2.putText(frame, f'{speed_kmh:.2f} km/h', (int(left), int(top - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            print(f'ID: {id}, New Car Detected')
        car_last_positions[id] = real_position

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False
    return True


# Sequence and camera information
sequence = 3  # S01, S03, S04
camera = 10  # c001, c002, c003, c004 ... c028
sequence_str = 'S{:02d}'.format(sequence)
camera_str = 'c{:03d}'.format(camera)

# Fire hydrant bounding box in the image
xtl = 375.01
ytl = 301.19
xbr = 406.51
ybr = 381.60

# Load calibration matrix
calibration_file = os.path.join(DATA_PATH, sequence_str, camera_str, 'calibration.txt')
calibration_mat = load_calibration_matrix(calibration_file)

# Calculate real-world coordinates of the fire hydrant bounding box
# Create the homogeneous coordinate [x, y, 1] for the top left and bottom right corners
top_left = np.array([xtl, ytl, 1])
bottom_right = np.array([xbr, ybr, 1])

# Apply the inverse calibration matrix to get real-world coordinates
calibration_inv = np.linalg.inv(calibration_mat)
real_world_top_left = np.dot(calibration_inv, top_left)
real_world_bottom_right = np.dot(calibration_inv, bottom_right)

# Calculate width and height in real-world coordinates
real_width = np.abs(real_world_bottom_right[0] - real_world_top_left[0])
real_height = np.abs(real_world_bottom_right[1] - real_world_top_left[1])

print("Fire Hydrant Bounding Box Width (meters):", real_width)
print("Fire Hydrant Bounding Box Height (meters):", real_height)

# Known length of hydrant
length_hydrant_meters = 1

# Conversion factor from pixels to meters
conversion_factor = (length_hydrant_meters / real_height)

print("Conversion Factor (meters per pixel):", conversion_factor)

# Load ground truth (gt) data
gt_file = os.path.join(DATA_PATH, sequence_str, camera_str, 'gt', 'gt.txt')
gt = np.loadtxt(gt_file, delimiter=',')

# Load video file
video_file = os.path.join(DATA_PATH, sequence_str, camera_str, 'vdo.avi')
cap = cv2.VideoCapture(video_file)

# Dictionary to store last positions of cars
car_last_positions = {}

# Dictionary to store speeds for each ID
speeds = {}

# Get FPS from the DATA_TREE
fps = DATA_TREE[sequence_str][camera_str]
print("FPS from DATA_TREE:", fps)

# # Function to get FPS from the video
# def get_fps(video_path):
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     cap.release()
#     return fps

# # Update FPS calculation from the video
# fps_video = get_fps(video_file)
# print("FPS from Video:", fps_video)

# Main loop to process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Call the function to visualize bounding box and calculate speed
    if not visualize_bbox_and_speed(fps, gt, calibration_inv, car_last_positions, conversion_factor, int(cap.get(cv2.CAP_PROP_POS_FRAMES)), frame, speeds):
        break

    # Slow down video display (10ms delay)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Calculate and print average speed for each ID
for id, speed_list in speeds.items():
    avg_speed = np.mean(speed_list)
    print(f'Average Speed for ID {id}: {avg_speed:.2f} km/h')
