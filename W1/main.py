import cv2
import numpy as np
import json

MEAN_STD_COMPUTED = True

# Load a video file with OpenCV
PATH_TO_DATA = './data/AICity_data/'
PATH_TO_VIDEO = f"{PATH_TO_DATA}train/S03/c010/vdo.avi"
PATH_TO_TMP = "./tmp/"
PATH_TO_OUTPUT = "./output/"

FRAMES_PERCENTAGE = 0.25
cap = cv2.VideoCapture(PATH_TO_VIDEO)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_dict = json.load(open(f"{PATH_TO_OUTPUT}frame_dict.json"))

if not MEAN_STD_COMPUTED:

    # Get the first 25% of the frames
    frames = []
    for i in range(int(total_frames * FRAMES_PERCENTAGE)):
        ret, frame = cap.read()
        # Set frames to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not ret:
            break
        frames.append(frame)

    # Convert the list to a numpy array
    frames = np.array(frames)

    # Compute the mean and standard deviation for each pixel
    mean = np.mean(frames, axis=0)
    std = np.std(frames, axis=0)

    # Save the mean and standard deviation
    np.save(f"{PATH_TO_TMP}mean.npy", mean)
    np.save(f"{PATH_TO_TMP}std.npy", std)
else:
    mean = np.load(f"{PATH_TO_TMP}mean.npy")
    std = np.load(f"{PATH_TO_TMP}std.npy")

# Truncate the mean and standard deviation to 0 and 255
mean_to_viz = np.clip(mean, 0, 255).astype(np.uint8)
std_to_viz = np.clip(std, 0, 255).astype(np.uint8)

# Save the mean and standard deviation as images
cv2.imwrite(f"{PATH_TO_OUTPUT}mean.png", mean_to_viz)
cv2.imwrite(f"{PATH_TO_OUTPUT}std.png", std_to_viz)

# Build a binary for each frame for the las 75% of the frames using the mean and standard deviation
binary_frames = []
ALPHA = 10
# Set cap initial position to 25% of the video
cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * FRAMES_PERCENTAGE))

for i in range(int(total_frames * FRAMES_PERCENTAGE), total_frames):
    ret, frame = cap.read()
    # Set frames to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not ret:
        break
    binary_frame = (frame - mean >= ALPHA * (std + 2)).astype(np.uint8)
    # Set binary frame to 255
    binary_frame = binary_frame * 255
    # Add rectangles to the frame
    for car_id, car in frame_dict[str(i)].items():
        x1 = int(car['xtl'])
        y1 = int(car['ytl'])
        x2 = int(car['xbr'])
        y2 = int(car['ybr'])
        cv2.rectangle(binary_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    # Show binary image
    cv2.imshow('binary_frame', binary_frame)
    cv2.waitKey(0)
    binary_frames.append(binary_frame)



