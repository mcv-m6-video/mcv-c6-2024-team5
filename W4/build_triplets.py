import os
import numpy as np
import cv2

DATA_PATH = 'data/aic19-track1-mtmc-train/train/'
# Sequence, camera and FPS
DATA_TREE = {
    'S01': {
        'c001': 10,
        'c002': 10,
        'c003': 10,
        'c004': 10,
        'c005': 10,
    },
    'S03': {
        'c010': 10,
        'c011': 10,
        'c012': 10,
        'c013': 10,
        'c014': 10,
        'c015': 8,
    },
    'S04': {
        'c016': 10,
        'c017': 10,
        'c018': 10,
        'c019': 10,
        'c020': 10,
        'c021': 10,
        'c022': 10,
        'c023': 10,
        'c024': 10,
        'c025': 10,
        'c026': 10,
        'c027': 10,
        'c028': 10
    }
}

def get_row_frame(gt, frame_to_find):
    for row in gt[gt[:, 0] == frame_to_find]:
        n_frame, id, left, top, width, height = row[:6].astype(int)

def get_row_id(gt, id_to_find):
    for row in gt[gt[:, 1] == id_to_find]:
        n_frame, id, left, top, width, height = row[:6].astype(int)

# GT format: frame, ID, left, top, width, height, 1, -1, -1, -1

def crop_and_save_image(video_path, frame, bbox, save_path):
    image = cv2.VideoCapture(video_path)
    image.set(1, frame)
    _, image = image.read()
    x, y, w, h = bbox
    if x < 0 or y < 0 or w < 0 or h < 0:
        return False
    if image is None:
        return False
    crop = image[y:y+h, x:x+w]
    cv2.imwrite(save_path, crop)
    return True

def generate_triplets(gt, sequence, num_triplets=100):
    # Directory for saving crops
    save_dir = os.path.join(DATA_PATH, sequence, 'triplets')
    os.makedirs(save_dir, exist_ok=True)

    triplets = []
    unique_ids = np.unique(gt[:, 1])
    i = 0

    while len(triplets) < num_triplets:
        # Randomly select an anchor ID
        anchor_id = np.random.choice(unique_ids)
        
        # Find all frames for this ID
        anchor_frames = gt[gt[:, 1] == anchor_id]
        
        # Randomly select two different frames for anchor and positive
        if len(anchor_frames) > 1:
            choices = np.random.choice(range(len(anchor_frames)), size=2, replace=False)
            anchor_frame_row = anchor_frames[choices[0], :]
            positive_frame_row = anchor_frames[choices[1], :]
        else:
            continue

        # Find a negative ID
        negative_ids = np.setdiff1d(unique_ids, np.array([anchor_id]))
        negative_id = np.random.choice(negative_ids)
        
        # Find frames for this negative ID
        negative_frames = gt[gt[:, 1] == negative_id]
        
        # Randomly select a frame for negative
        negative_frame_row = negative_frames[np.random.choice(range(len(negative_frames))), :]

        # Crop and save anchor, positive, negative images
        for idx, row in enumerate([anchor_frame_row, positive_frame_row, negative_frame_row]):
            frame, id, left, top, width, height = row[:6].astype(int)
            camera_str = 'c{:03d}'.format(int(row[-1]))  # Assuming the camera ID is stored in the last column of gt
            video_path = os.path.join(DATA_PATH, sequence, camera_str, 'vdo.avi')
            bbox = [left, top, width, height]
            save_path = os.path.join(save_dir, f'triplet_{i}_{["anchor", "positive", "negative"][idx]}.jpg')
            success = crop_and_save_image(video_path, frame, bbox, save_path)
            if not success:
                continue

        triplets.append((anchor_frame_row[0], positive_frame_row[0], negative_frame_row[0]))
        i += 1

# Assuming you've stacked all GTs for each sequence in a single array, here's how you'd call the function
for sequence_str in DATA_TREE.keys():
    gt_combined = None  # Placeholder for combined GT arrays
    for camera_str in DATA_TREE[sequence_str].keys():
        camera_path = os.path.join(DATA_PATH, sequence_str, camera_str)
        gt_path = os.path.join(camera_path, 'gt/gt.txt')
        
        # Load and append GT data
        gt = np.loadtxt(gt_path, delimiter=',')
        # gt = np.append(gt, np.full((gt.shape[0], 1), int(camera_str[1:]), axis=1))  # Append camera ID to GT
        gt_camera_column = np.full((gt.shape[0], 1), int(camera_str[1:]))  # Create a column for camera IDs
        gt = np.concatenate((gt, gt_camera_column), axis=1)  # Append the camera ID column to GT
        if gt_combined is None:
            gt_combined = gt
        else:
            gt_combined = np.vstack((gt_combined, gt))
    
    # Now gt_combined contains all GTs for the sequence
    generate_triplets(gt_combined, sequence_str, 100)
