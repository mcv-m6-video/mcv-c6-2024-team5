import numpy as np
import os
import cv2
import numpy as np

# Load the images
flow_x_path = 'flow/flow_x_00005.jpg'
flow_y_path = 'flow/flow_y_00005.jpg'

flow_x_img = cv2.imread(flow_x_path, cv2.IMREAD_UNCHANGED)
flow_y_img = cv2.imread(flow_y_path, cv2.IMREAD_UNCHANGED)

# Convert images to numpy arrays
flow_x = np.array(flow_x_img)
flow_y = np.array(flow_y_img)

# Ensure that the shapes of the flow images match
assert flow_x.shape == flow_y.shape, "The dimensions of the two images do not match."

# Generate a matrix to store the flow range -1 to 1
flow_rgb = np.zeros((flow_x.shape[0], flow_x.shape[1], 3), dtype=np.float32)

# Assign the flow_x and flow_y to the red and green channels
# flow_rgb[:, :, 0] = flow_x  # Red channel
# flow_rgb[:, :, 1] = flow_y  # Green channel
# Considering flow_x and flow_y are in the range of 0-255, we need to normalize them to be in the range of -1 to 1
flow_rgb[:, :, 0] = (flow_x - 127.5) / 127.5
flow_rgb[:, :, 1] = (flow_y - 127.5) / 127.5

# Max and min values of the flow
print(max(flow_x.flatten()))
print(min(flow_x.flatten()))
# Max and min values of the flow normalized to -1 to 1
print(max(flow_rgb[:, :, 0].flatten()))
print(min(flow_rgb[:, :, 0].flatten()))

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
hsv[..., 2] = 255  # Set value to maximum for full brightness
rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Save the image
cv2.imwrite(f'flow/gt_flow.png', rgb)