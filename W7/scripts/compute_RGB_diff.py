import os
import cv2

data_path = '../data/frames/'
video_path = data_path + 'jump/Goalkeeper_Felix_Schwake_jump_f_cm_np1_ri_bad_0/'

# RGB images are the iamges finished in .jpg and start with frame
rgb_images = [f for f in os.listdir(video_path) if f.endswith('.jpg') and f.startswith('frame')]

# Compute RGB difference and save the images as rgb_diff_{frame_number}.jpg
# If there are 10 images we have from frame00001.jpg to frame00010.jpg, we will compute the difference between frame00002.jpg and frame00001.jpg and save it as rgb_diff_00001.jpg
for i in range(1, len(rgb_images)+1):
    # Load the current frame
    current_frame = cv2.imread(video_path + f'frame{i:05d}.jpg')
    # Load the previous frame
    future_frame = cv2.imread(video_path + f'frame{i+1:05d}.jpg')
    
    # Compute the difference between the two frames
    rgb_diff = cv2.absdiff(current_frame, future_frame)
    
    # Save the difference image
    cv2.imwrite(video_path + f'rgb_diff_{i:05d}.jpg', rgb_diff)