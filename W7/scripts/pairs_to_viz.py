import numpy as np
import os
import cv2
import numpy as np

root_path = '../data'

# In data there are two folders: frames and rawframes with the same directory structure
# i.e data/frames/brush_hair/{video_name}/frame00001.jpg and data/rawframes/brush_hair/{video_name}/flow_x_00001.jpg and data/rawframes/brush_hair/{video_name}/flow_y_00001.jpg
# We'll get the flow images from the rawframes folder and will save the visualization in the frames folder

# Iterate over the folders in the rawframes directory
for folder in os.listdir(f'{root_path}/rawframes'):
    # Iterate over the videos in the folder
    for video in os.listdir(f'{root_path}/rawframes/{folder}'):
        # Iterate over the flow images in the video directory
        for flow_x in os.listdir(f'{root_path}/rawframes/{folder}/{video}'):
            # Check if the file is a flow_x image
            if 'flow_x' in flow_x:
                # Load the flow_x image
                flow_x_img = cv2.imread(f'{root_path}/rawframes/{folder}/{video}/{flow_x}', cv2.IMREAD_UNCHANGED)
                # Load the corresponding flow_y image
                flow_y_img = cv2.imread(f'{root_path}/rawframes/{folder}/{video}/flow_y{flow_x[-10:-4]}.jpg', cv2.IMREAD_UNCHANGED)
                print(f'{root_path}/rawframes/{folder}/{video}/{flow_x}')
                print(f'{root_path}/rawframes/{folder}/{video}/flow_y{flow_x[-10:-4]}.jpg')
                # Extract frame number from flow_x filename
                frame_number = flow_x[-10:-4]
                print(frame_number)

                # Convert images to numpy arrays
                flow_x = np.array(flow_x_img)
                flow_y = np.array(flow_y_img)
                
                # Ensure that the shapes of the flow images match
                assert flow_x.shape == flow_y.shape, "The dimensions of the two images do not match."
                
                # Generate a matrix to store the flow range -1 to 1
                flow_rgb = np.zeros((flow_x.shape[0], flow_x.shape[1], 3), dtype=np.float32)
                
                # Considering flow_x and flow_y are in the range of 0-255, we need to normalize them to be in the range of -1 to 1
                flow_rgb[:, :, 0] = (flow_x - 127.5) / 127.5
                flow_rgb[:, :, 1] = (flow_y - 127.5) / 127.5
                
                # Split the flow into u and v components
                u = flow_rgb[:,:,0].astype(float)
                v = flow_rgb[:,:,1].astype(float) 
                valid = flow_rgb[:,:,2].astype(bool)
                
                # Create HSV and RGB visualizations for the flow
                hsv = np.zeros((flow_rgb.shape[0], flow_rgb.shape[1], 3), dtype=np.uint8)
                mag, ang = cv2.cartToPolar(u, v)
                # Old: Black for non moving pixels
                # hsv[..., 1] = 255
                # hsv[..., 0] = ang * 180 / np.pi / 2
                # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                # New: White for non moving pixels
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                hsv[..., 2] = 255
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                
                # Save the image with the correct frame number
                cv2.imwrite(f'{root_path}/frames/{folder}/{video}/flow{frame_number}.png', rgb)
                print(f'Saved: {root_path}/frames/{folder}/{video}/flow{frame_number}.png')

                # # Save the image
                # cv2.imwrite(f'{root_path}/frames/{folder}/{video}/flow_{flow_x[-10:-4]}.png', rgb)
                # print (f'{root_path}/frames/{folder}/{video}/flow_{flow_x[-10:-4]}.png')