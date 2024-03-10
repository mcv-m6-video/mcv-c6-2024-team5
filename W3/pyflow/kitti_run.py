# Updated for use with Docker container and images in /data folder

import numpy as np
from PIL import Image
import time
import argparse
import pyflow
import cv2

def save_flow_to_image(u, v, valid, filename):
    assert u.shape == v.shape == valid.shape, "Mismatch in dimension of flow components and valid mask"
    
    # Initialize the flow image with all invalid pixels
    flow_img = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.uint16)
    
    # Set the flow values, scaling and encoding as per KITTI format
    flow_img[..., 0] = (u * 64.0 + 2**15).astype(np.uint16)
    flow_img[..., 1] = (v * 64.0 + 2**15).astype(np.uint16)
    flow_img[..., 2] = valid
    
    flow_img_bgr = np.stack((flow_img[:, :, 2], flow_img[:, :, 1], flow_img[:, :, 0]), axis=-1)
    # Save the image
    cv2.imwrite(filename, flow_img_bgr)

# Setup argument parser
parser = argparse.ArgumentParser(description='Compute Coarse2Fine Optical Flow on images in /data directory')
parser.add_argument('--image1', type=str, help='Path to the first image', default='./data/000045_10.png')
parser.add_argument('--image2', type=str, help='Path to the second image', default='./data/000045_11.png')
parser.add_argument('--output_flow', type=str, help='Path to save the flow visualization', default='./data/flow.png')
parser.add_argument('--output_warped', type=str, help='Path to save the warped second image', default='./data/warped.png')
parser.add_argument('--output_raw_flow', type=str, help='Path to save the raw flow as numpy file', default='./data/raw_flow.png')
parser.add_argument('--viz', action='store_true', help='Visualize (i.e., save) output of flow.', default=True)

args = parser.parse_args()

# Load images
# Load images and convert them to float
im1 = np.array(Image.open(args.image1)).astype(float) / 255.
im2 = np.array(Image.open(args.image2)).astype(float) / 255.

# Check if the images are grayscale (2D) and add an extra dimension if they are
if im1.ndim == 2:
    im1 = im1[:, :, np.newaxis]
if im2.ndim == 2:
    im2 = im2[:, :, np.newaxis]

# Flow Options
alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

# Calculate flow
s = time.time()
u, v, im2W = pyflow.coarse2fine_flow(im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations, colType)
e = time.time()
print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (e - s, im1.shape[0], im1.shape[1], im1.shape[2]))

# Save flow and warped image if visualization is requested
if args.viz:

    # Create HSV and RGB visualizations for the flow
    hsv = np.zeros((im1.shape[0], im1.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(u, v)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Save output images
    valid = np.ones(u.shape, dtype=np.uint8)  # Assuming all pixels are valid for the example
    save_flow_to_image(u, v, valid, args.output_raw_flow)
    cv2.imwrite(args.output_flow, rgb)
    cv2.imwrite(args.output_warped, (im2W[:, :, 0] * 255).astype(np.uint8)) # Assuming im2W is also grayscale
