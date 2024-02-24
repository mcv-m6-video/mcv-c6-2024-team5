import cv2
import numpy as np

MEAN_STD_COMPUTED = True

# Load a video file with OpenCV
path_to_data = './data/AICity_data/'
path_to_video = f"{path_to_data}train/S03/c010/vdo.avi"
path_to_tmp = "./tmp/"
path_to_output = "./output/"

cap = cv2.VideoCapture(path_to_video)
frames_percentage = 0.25
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if not MEAN_STD_COMPUTED:

    # Get the first 25% of the frames
    frames = []
    for i in range(int(total_frames * frames_percentage)):
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
    np.save(f"{path_to_tmp}mean.npy", mean)
    np.save(f"{path_to_tmp}std.npy", std)
else:
    mean = np.load(f"{path_to_tmp}mean.npy")
    std = np.load(f"{path_to_tmp}std.npy")

# Truncate the mean and standard deviation to 0 and 255
mean_to_viz = np.clip(mean, 0, 255).astype(np.uint8)
std_to_viz = np.clip(std, 0, 255).astype(np.uint8)

# Save the mean and standard deviation as images
cv2.imwrite(f"{path_to_output}mean.png", mean_to_viz)
cv2.imwrite(f"{path_to_output}std.png", std_to_viz)

# Build a binary for each frame for the las 75% of the frames using the mean and standard deviation
binary_frames = []
ALPHA = 10
# Set cap initial position to 25% of the video
cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * frames_percentage))

for i in range(int(total_frames * frames_percentage), total_frames):
    ret, frame = cap.read()
    # Set frames to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not ret:
        break
    binary_frame = (frame - mean >= ALPHA * (std + 2)).astype(np.uint8)
    # Set binary frame to 255
    binary_frame = binary_frame * 255
    # Show binary image
    cv2.imshow('binary_frame', binary_frame)
    cv2.waitKey(0)
    binary_frames.append(binary_frame)



