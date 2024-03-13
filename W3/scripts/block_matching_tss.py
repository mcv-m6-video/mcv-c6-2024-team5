import cv2
import numpy as np
import time
import argparse

# Reference: https://github.com/gautamo/BlockMatching/tree/master

def transform_kitti_flow_to_image(flow_dir, image_dir):
    """
    Transform KITTI optical flow GT to a visualization image
    :param flow_dir: Path to the KITTI optical flow .png file
    :param image_dir: Path to the corresponding image for visualization
    :param output_dir: Output directory to save the visualization image
    """
    # Read flow and image
    flow = cv2.imread(flow_dir, cv2.IMREAD_UNCHANGED)
    image = cv2.imread(image_dir, cv2.IMREAD_UNCHANGED)

    # Split the flow into u and v components
    u = (flow[:, :, 2].astype(float) - 2 ** 15) / 64.0
    v = (flow[:, :, 1].astype(float) - 2 ** 15) / 64.0
    valid = flow[:, :, 0].astype(bool)

    # Create HSV and RGB visualizations for the flow
    hsv = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(u, v)
    # Old: Black for non-moving pixels
    # hsv[..., 1] = 255
    # hsv[..., 0] = ang * 180 / np.pi / 2
    # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # New: White for non-moving pixels
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = 255  # Set value to maximum for full brightness
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Save the image
    filename = ('gt_flow_visualization.png')
    cv2.imwrite(filename, rgb)

# Function to get the block searched in the anchor search area
def getBlockZone(p, aSearch, tBlock, blockSize):
    px, py = p  # coordinates of macroblock center
    px, py = px - int(blockSize / 2), py - int(blockSize / 2)  # get top left corner of macroblock

    # Padding to handle boundary cases
    px_start, px_end = max(0, px), min(aSearch.shape[1], px + blockSize)
    py_start, py_end = max(0, py), min(aSearch.shape[0], py + blockSize)

    # Get the block from the anchor search area
    aBlock = aSearch[py_start:py_end, px_start:px_end]

    # If the extracted block is smaller than (blockSize, blockSize), pad it with zeros
    if aBlock.shape[0] < blockSize or aBlock.shape[1] < blockSize:
        pad_x = max(blockSize - aBlock.shape[1], 0)
        pad_y = max(blockSize - aBlock.shape[0], 0)
        aBlock = np.pad(aBlock, ((0, pad_y), (0, pad_x)), mode='constant')

    try:
        assert aBlock.shape == tBlock.shape  # must be same shape

    except Exception as e:
        print(e)
        print(f"ERROR - ABLOCK SHAPE: {aBlock.shape} != TBLOCK SHAPE: {tBlock.shape}")

    return aBlock


# Function to calculate the error between two blocks
def get_error(block1, block2, error_function):
    if error_function == 'SAD':
        # pad the blocks to match the block size
        if block1.shape[0] != block2.shape[0] or block1.shape[1] != block2.shape[1]:
            block1 = np.pad(block1, ((block2.shape[0] - block1.shape[0], 0), (block2.shape[1] - block1.shape[1], 0)), 'constant')
        return np.sum(np.abs(block1 - block2))
    elif error_function == 'SSD':
        return np.sum(np.square(block1 - block2))
    elif error_function == 'CCOEFF':
        # Cross-correlation coefficient
        mean_block1 = np.mean(block1)
        mean_block2 = np.mean(block2)
        num = np.sum((block1 - mean_block1) * (block2 - mean_block2))
        den = np.sqrt(np.sum(np.square(block1 - mean_block1)) * np.sum(np.square(block2 - mean_block2)))
        if den == 0:
            return np.inf  # Return infinity for very large error
        else:
            return -num / den  # Negative value for minimizing
    elif error_function == 'NCC':
        # Normalized cross-correlation
        mean_block1 = np.mean(block1)
        mean_block2 = np.mean(block2)
        num = np.sum((block1 - mean_block1) * (block2 - mean_block2))
        den = np.sqrt(np.sum(np.square(block1 - mean_block1)) * np.sum(np.square(block2 - mean_block2)))
        if den == 0:
            return np.inf  # Return infinity for very large error
        else:
            return -num / den
    else:
        raise ValueError("Unknown error function. Use 'SAD', 'SSD', 'CCOEFF', or 'NCC'.")


# Function to find the motion vector between two blocks by subtracting block centers
def find_motion_vector(original_coord, best_match_coord):
    y1, x1 = original_coord
    x2, y2 = best_match_coord
    dx = x2 - x1
    dy = y2 - y1
    return dx, dy


# Function to read optical flow from a PNG file
def read_flow_png(filename):
    flow_img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    flow_u = (flow_img[:, :, 2].astype(np.float32) - 2**15) / 64.0
    flow_v = (flow_img[:, :, 1].astype(np.float32) - 2**15) / 64.0
    valid = flow_img[:, :, 0].astype(bool)
    return flow_u, flow_v, valid


# Function to compute MSEN (Mean Squared Error of Normals) and PEPN (Percentage of Erroneous Pixels)
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


# Function to visualize the comparison between blocks
def visualize_comparison(img1, block1, coord1, block2, coord2, search_window_size):
    x1, y1 = coord1
    x2, y2 = coord2

    img_copy = img1.copy()
    h, w = block1.shape

    # Draw rectangles for the blocks
    cv2.rectangle(img_copy, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)  # Green (Reference block)
    cv2.rectangle(img_copy, (x2, y2), (x2 + w, y2 + h), (0, 0, 255), 2)  # Red (Matching block)

    # Draw search window
    cv2.rectangle(img_copy, (x1 - search_window_size // 2, y1 - search_window_size // 2),
                  (x1 + search_window_size // 2 + block1.shape[1], y1 + search_window_size // 2 + block1.shape[0]), (255, 0, 0), 2)  # Blue

    cv2.imshow('Comparison', img_copy)
    cv2.waitKey(1)


# Function to find the best match using Three-Step Search
def get_best_match(reference_block, img, search_area_x1, search_area_y1, search_area_x2, search_area_y2, block_size, error_function, img_ori):
    step = 4
    search_area_h, search_area_w = search_area_y2 - search_area_y1, search_area_x2 - search_area_x1
    # get center of search area
    acx, acy = search_area_x1 + search_area_w // 2, search_area_y1 + search_area_h // 2
    acx_orig, acy_orig = acx, acy

    # print('center:', acx, acy)

    min_error = float("inf")
    min_p = (acx, acy)
    best_block_coords = []

    while step >= 1:
        # ensure that point_list is within the search area
        point_list = [(acx, acy), (acx - step, acy), (acx + step, acy), (acx, acy - step), (acx, acy + step),
                        (acx - step, acy - step), (acx + step, acy - step), (acx - step, acy + step), (acx + step, acy + step)]
        
        
        for p in point_list:

            px, py = p
   
            search_block = img_ori[py-block_size//2:py+block_size//2, px-block_size//2:px+block_size//2]

            #if search block is smaller than block size, pad it with zeros
            if search_block.shape[0] < block_size or search_block.shape[1] < block_size:
                pad_x = max(block_size - search_block.shape[1], 0)
                pad_y = max(block_size - search_block.shape[0], 0)
                search_block = np.pad(search_block, ((0, pad_y), (0, pad_x)), mode='constant')

            error = get_error(reference_block, search_block, error_function)  # determine error
            if error < min_error:  # store point with minimum error
                min_error = error
                min_p = p

            # Append all block coordinates for visualization
            best_block_coords.append((px, py))
    
        acx, acy = min_p  # update anchor center with minimum error
        if acx<0 and acy<0:
            print('Negative coordinates:', acx, acy)
        step = int(step / 2)

    # cv2.rectangle(img_ori,(acx - block_size // 2, acy - block_size // 2), (acx + block_size // 2, acy + block_size // 2), 0, 1)  # Red (Matching block
    # cv2.rectangle(img_ori, (acx_orig - block_size // 2, acy_orig - block_size // 2), (acx_orig + block_size // 2, acy_orig + block_size // 2), 170, 1)  # Green (Reference block)
    # cv2.imshow('Search Block', img_ori)
    # cv2.waitKey(0)

    # Calculate the global coordinates of the best match block
    # print('best match:', min_p)
    return [acx,acy],[acx_orig,acy_orig], best_block_coords


# Function for optical flow estimation using Three-Step Search
def optical_flow_estimation_tss(img1_path, img2_path, block_size, search_window_size, step_size, error_function,
                                motion_direction="forward", visualize=False):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    img_ori = img1.copy()

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)

    # Uncomment to apply pre-processing    
    # img1 = cv2.GaussianBlur(img1, (5, 5), 0)
    # img2 = cv2.GaussianBlur(img2, (5, 5), 0)
    # img1 = cv2.medianBlur(img1, 5)
    # img2 = cv2.medianBlur(img2, 5)
    # img1 = cv2.equalizeHist(img1)
    # img2 = cv2.equalizeHist(img2)
    # img1 = cv2.Canny(img1, 100, 200)
    # img2 = cv2.Canny(img2, 100, 200)

    h, w = img1.shape
    pred_flow_u = np.zeros(np.shape(img1))
    pred_flow_v = np.zeros(np.shape(img1))

    start_time = time.time()

    for y in range(search_window_size//2, h-search_window_size//2, step_size):
        for x in range(search_window_size//2, w-search_window_size//2, step_size):
            reference_block = img2[max(0, y - block_size // 2):min(h, y + block_size // 2),max(0, x - block_size // 2):min(w, x + block_size // 2)]
            search_area_x1 =  max(0, x - search_window_size // 2)
            search_area_x2 = min(w, x + search_window_size // 2)
            search_area_y1 = max(0, y - search_window_size // 2)
            search_area_y2 = min(h, y + search_window_size // 2)

            # visualize the block and the search area on img2 always
            # cv2.rectangle(img_ori, (x - block_size // 2, y - block_size // 2), (x + block_size // 2, y + block_size // 2), (0, 255, 0), 1)  # Green (Reference block)
            # cv2.rectangle(img_ori, (search_area_x1, search_area_y1), (search_area_x2, search_area_y2), (255, 0, 0), 1)  # Blue (Search area)

            # cv2.imshow('Search Area', img_ori)
            # cv2.waitKey(0)
            # Get the best match block using the Three-Step Search within this search area
            if motion_direction == "forward":
                # Forward motion: Searching for best match block in img1
                best_match_coord, reference_coord, _ = get_best_match(reference_block, img2, search_area_x1, search_area_y1,
                                                    search_area_x2, search_area_y2, block_size, error_function, img_ori)
            else:  # Backward motion
                # Backward motion: Searching for best match block in img2
                best_match_coord, reference_coord, _ = get_best_match(reference_block, img2, search_area_x1, search_area_y1,
                                                    search_area_x2, search_area_y2, block_size, error_function, img_ori)
            
            
            # Get the actual block from img1 or img2 based on motion direction
            match_block = getBlockZone(best_match_coord, img1 if motion_direction == "forward" else img2, reference_block, block_size)
            
            # Calculate the displacement (motion vector) 
            dx,dy = reference_coord[0] - best_match_coord[0], reference_coord[1] - best_match_coord[1]
            # print('Displacement:', dx, dy)

            # Update pred_flow_u and pred_flow_v
            pred_flow_u[y-block_size//2:y+block_size//2, x-block_size//2:x+block_size//2] = dx
            pred_flow_v[y-block_size//2:y+block_size//2, x-block_size//2:x+block_size//2] = dy

            if visualize:
                visualize_comparison(img_ori, reference_block, (x, y), match_block, best_match_coord, search_window_size)

    runtime = time.time() - start_time
    print(f"Total runtime: {runtime} seconds")

    # # Uncomment to apply post-processing
    magnitude = np.sqrt(pred_flow_u ** 2 + pred_flow_v ** 2)
    threshold = 3  # Adjust threshold as needed
    mask = magnitude > threshold
    pred_flow_u[~mask] = 0
    pred_flow_v[~mask] = 0

    return pred_flow_u, pred_flow_v, runtime


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optical Flow Estimation")
    parser.add_argument("--img1", type=str, help="Path to the first image")
    parser.add_argument("--img2", type=str, help="Path to the second image")
    parser.add_argument("--gt_flow", type=str, help="Path to the ground truth flow")
    parser.add_argument("--visualize", action='store_true', help="Visualize the comparison")
    parser.add_argument("--motion_direction", type=str, default="forward", choices=["forward", "backward"],
    help="Direction of motion estimation (forward or backward)")
    args = parser.parse_args()

    # Define a filename for the output results
    output_filename = "optical_flow_results.txt"

    # Run optical flow with different parameters
    idx = 0
    with open(output_filename, "w") as output_file:
        for area_of_search_size in [4, 8, 16]:
            for block_size in [4, 8, 16]:
                for step_size in [1, 2, 4]:
                    for error_function in ["NCC", "SAD", "SSD", "CCOEFF"]:
                        for tau in [3, 5, 8, 10]:
                            output_file.write('-' * 40 + "\n")
                            output_file.write("Combination:\n")
                            output_file.write(f"Area Size: {area_of_search_size}, Block Size: {block_size}, Step Size: {step_size}, Error Function: {error_function}, Tau: {tau}\n")
                            print("Combination:")
                            print(f"Area Size: {area_of_search_size}, Block Size: {block_size}, Step Size: {step_size}, Error Function: {error_function}, Tau: {tau}")

                            # Estimate Optical Flow
                            pred_flow_u, pred_flow_v, runtime = optical_flow_estimation_tss(args.img1, args.img2, block_size, area_of_search_size,
                                                                                 step_size, error_function, args.visualize)

                            # Read Ground Truth Flow
                            gt_flow_u, gt_flow_v, valid = read_flow_png(args.gt_flow)

                            # Compute MSEN and PEPN
                            mse, pepn = compute_msen_pepn(pred_flow_u, pred_flow_v, gt_flow_u, gt_flow_v, valid, tau)

                            output_file.write("MSEN: " + str(mse) + "\n")
                            output_file.write("PEPN: " + str(pepn) + "\n")
                            output_file.write("Runtime: " + str(runtime) + " seconds\n\n")
                            print("MSEN:", mse)
                            print("PEPN:", pepn)
                            print("Runtime:", runtime, "seconds")
                            idx += 1

    print("Results saved to:", output_filename)


#     # save flow to image
#     save_flow_to_image(pred_flow_u, pred_flow_v, np.ones_like(pred_flow_v), 'flow.png')

#     # transform from kitti flow
#     flow_dir = 'flow.png'
#     image_dir = args.img1
#     transform_kitti_flow_to_image(flow_dir, image_dir)