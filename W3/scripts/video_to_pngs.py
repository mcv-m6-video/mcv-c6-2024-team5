# Convert video from .avi to a series of ordered .png images
import cv2
import os

# Path to the video file
video_path = '../data/AICity_data/train/S03/c010/vdo.avi'
# Path to the directory to save the images
output_dir = '../data/AICity_data/train/S03/c010/frames'

# Create the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Read the first frame
ret, frame = cap.read()
frame_num = 0
while ret:
    # Save the frame as a .png image
    cv2.imwrite(os.path.join(output_dir, f'{frame_num:06d}.png'), frame)
    frame_num += 1
    # Read the next frame
    ret, frame = cap.read()

# Release the video capture object
cap.release()

# Print the number of frames saved
print(f'Saved {frame_num} frames')