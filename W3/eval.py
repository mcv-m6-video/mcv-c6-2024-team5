import numpy as np
import cv2
import argparse

def read_flow_png(filename):
    flow_img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    flow_u = (flow_img[:, :, 2].astype(np.float32) - 2**15) / 64.0
    flow_v = (flow_img[:, :, 1].astype(np.float32) - 2**15) / 64.0
    valid = flow_img[:, :, 0].astype(bool)
    return flow_u, flow_v, valid

def compute_msen_pepn(pred_flow_u, pred_flow_v, gt_flow_u, gt_flow_v, valid, tau=3):
    error = np.sqrt((pred_flow_u - gt_flow_u)**2 + (pred_flow_v - gt_flow_v)**2)
    
    # Only consider non-occluded areas
    error_valid = error[valid]
    
    # Check if there are any non-occluded pixels
    if error_valid.size == 0:
        mse = np.nan
        pepn = np.nan
    else:
        mse = np.mean(error_valid)
        pepn = np.mean(error_valid > tau) * 100
    
    return mse, pepn

# Parse arguments
parser = argparse.ArgumentParser(description='Compute MSEN and PEPN for Optical Flow')
parser.add_argument('--predicted_flow', type=str, help='Path to the predicted flow .png file', default='./pyflow/local_data/raw_flow.png')
parser.add_argument('--ground_truth_flow', type=str, help='Path to the ground truth flow .png file', default='./data/training/flow_noc/000045_10.png')
args = parser.parse_args()

# Load predicted flow
pred_flow_u, pred_flow_v, pred_valid = read_flow_png(args.predicted_flow)

# Load ground truth flow and valid mask
gt_flow_u, gt_flow_v, gt_valid = read_flow_png(args.ground_truth_flow)

# Valid mask should be the logical AND between predicted and ground truth valid masks
valid = pred_valid & gt_valid

# Compute MSEN and PEPN
mse, pepn = compute_msen_pepn(pred_flow_u, pred_flow_v, gt_flow_u, gt_flow_v, valid)
print(f'MSEN: {mse:.4f}, PEPN: {pepn:.2f}%')
