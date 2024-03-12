# Transform KITTI optical flow GT to a visualization image

import os
import cv2
import numpy as np

image_dir = '../data/training/image_0/000045_10.png'
flow_dir = '../data/training/flow_noc/000045_10.png'
output_dir = './out'

flow = cv2.imread(flow_dir, cv2.IMREAD_UNCHANGED)
im1 = cv2.imread(image_dir, cv2.IMREAD_UNCHANGED)

# Split the flow into u and v components
u = (flow[:,:,2].astype(float) - 2**15) / 64.0
v = (flow[:,:,1].astype(float) - 2**15) / 64.0
valid = flow[:,:,0].astype(bool)

# Create HSV and RGB visualizations for the flow
hsv = np.zeros((im1.shape[0], im1.shape[1], 3), dtype=np.uint8)
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
cv2.imwrite(f'{output_dir}/gt_flow.png', rgb)