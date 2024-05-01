import os
import cv2
import pickle

data_path = '../data/frames/'
video_path = data_path + 'jump/Goalkeeper_Felix_Schwake_jump_f_cm_np1_ri_bad_0/'

frame_dir = 'Goalkeeper_Felix_Schwake_jump_f_cm_np1_ri_bad_0'
skeletons_pkl_path = '../data/hmdb51_2d.pkl'
skeletons = pickle.load(open(skeletons_pkl_path, 'rb'))

# Each pickle file corresponds to an action recognition dataset. The content of a pickle file is a dictionary with two fields: split and annotations

# Split: The value of the split field is a dictionary: the keys are the split names, while the values are lists of video identifiers that belong to the specific clip.
# Annotations: The value of the annotations field is a list of skeleton annotations, each skeleton annotation is a dictionary, containing the following fields:
# frame_dir (str): The identifier of the corresponding video.
# total_frames (int): The number of frames in this video.
# img_shape (tuple[int]): The shape of a video frame, a tuple with two elements, in the format of (height, width). Only required for 2D skeletons.
# original_shape (tuple[int]): Same as img_shape.
# label (int): The action label.
# keypoint (np.ndarray, with shape [M x T x V x C]): The keypoint annotation. M: number of persons; T: number of frames (same as total_frames); V: number of keypoints (25 for NTURGB+D 3D skeleton, 17 for CoCo, 18 for OpenPose, etc. ); C: number of dimensions for keypoint coordinates (C=2 for 2D keypoint, C=3 for 3D keypoint).
# keypoint_score (np.ndarray, with shape [M x T x V]): The confidence score of keypoints. Only required for 2D skeletons.

# Get the skeleton from frame_dir, so skeletons[annotations][frame_dir] where frame_dir = 'Goalkeeper_Felix_Schwake_jump_f_cm_np1_ri_bad_0'
for skeleton in skeletons['annotations']:
    if skeleton['frame_dir'] == frame_dir:
        vid_skeletons = skeleton['keypoint']
        break
print(vid_skeletons)

# RGB images are the iamges finished in .jpg and start with frame
rgb_images = [f for f in os.listdir(video_path) if f.endswith('.jpg') and f.startswith('frame')]

# Compute skeletons difference and save the images as skeletons_{frame_number}.jpg
# If there are 10 images we have from frame00001.jpg to frame00010.jpg, we will compute the skeleton from frame00001.jpg and save it as skeletons_00001.jpg
for i in range(1, len(rgb_images)+1):
    # Load the current frame
    current_frame = cv2.imread(video_path + f'frame{i:05d}.jpg')
    
    # Considering vid_skeletons is shape [M x T x V x C]): The keypoint annotation. M: number of persons; T: number of frames (same as total_frames); V: number of keypoints (25 for NTURGB+D 3D skeleton, 17 for CoCo, 18 for OpenPose, etc. ); C: number of dimensions for keypoint coordinates (C=2 for 2D keypoint, C=3 for 3D keypoint).

    # Add the skeleton to the current frame
    for person in range(vid_skeletons.shape[0]):
        for keypoint in range(vid_skeletons.shape[2]):
            x, y = vid_skeletons[person, i-1, keypoint]
            # Connect the keypoints, order is  [ "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle" ]
            if keypoint == 0:
                nose = (int(x), int(y))
            elif keypoint == 1:
                left_eye = (int(x), int(y))
            elif keypoint == 2:
                right_eye = (int(x), int(y))
            elif keypoint == 3:
                left_ear = (int(x), int(y))
            elif keypoint == 4:
                right_ear = (int(x), int(y))
            elif keypoint == 5:
                left_shoulder = (int(x), int(y))
            elif keypoint == 6:
                right_shoulder = (int(x), int(y))
            elif keypoint == 7:
                left_elbow = (int(x), int(y))
            elif keypoint == 8:
                right_elbow = (int(x), int(y))
            elif keypoint == 9:
                left_wrist = (int(x), int(y))
            elif keypoint == 10:
                right_wrist = (int(x), int(y))
            elif keypoint == 11:
                left_hip = (int(x), int(y))
            elif keypoint == 12:
                right_hip = (int(x), int(y))
            elif keypoint == 13:
                left_knee = (int(x), int(y))
            elif keypoint == 14:
                right_knee = (int(x), int(y))
            elif keypoint == 15:
                left_ankle = (int(x), int(y))
            elif keypoint == 16:
                right_ankle = (int(x), int(y))
            else:
                continue
        line_width = 1
        point_radius = 3
        ## Draw the skeleton
        # HEAD
        color = (0, 255, 0)
        cv2.circle(current_frame, nose, point_radius, color, -1)
        cv2.circle(current_frame, left_eye, point_radius, color, -1)
        cv2.circle(current_frame, right_eye, point_radius, color, -1)
        cv2.circle(current_frame, left_ear, point_radius, color, -1)
        cv2.circle(current_frame, right_ear, point_radius, color, -1)
        cv2.line(current_frame, nose, left_eye, color, line_width)
        cv2.line(current_frame, nose, right_eye, color, line_width)
        cv2.line(current_frame, left_eye, left_ear, color, line_width)
        cv2.line(current_frame, right_eye, right_ear, color, line_width)
        cv2.line(current_frame, left_eye, left_shoulder, color, line_width)
        cv2.line(current_frame, right_eye, right_shoulder, color, line_width)
        ## BODY
        color = (255, 0, 0)
        cv2.circle(current_frame, left_shoulder, point_radius, color, -1)
        cv2.circle(current_frame, right_shoulder, point_radius, color, -1)
        cv2.circle(current_frame, left_elbow, point_radius, color, -1)
        cv2.circle(current_frame, right_elbow, point_radius, color, -1)
        cv2.circle(current_frame, left_wrist, point_radius, color, -1)
        cv2.circle(current_frame, right_wrist, point_radius, color, -1)
        cv2.line(current_frame, left_shoulder, right_shoulder, color, line_width)
        cv2.line(current_frame, left_shoulder, left_elbow, color, line_width)
        cv2.line(current_frame, right_shoulder, right_elbow, color, line_width)
        cv2.line(current_frame, left_elbow, left_wrist, color, line_width)
        cv2.line(current_frame, right_elbow, right_wrist, color, line_width)
        ## BODY-LEG CONNECTION
        color = (255, 0, 255)
        cv2.line(current_frame, left_shoulder, left_hip, color, line_width)
        cv2.line(current_frame, right_shoulder, right_hip, color, line_width)
        ## LEGS
        color = (0, 165, 255)
        cv2.circle(current_frame, left_hip, point_radius, color, -1)
        cv2.circle(current_frame, right_hip, point_radius, color, -1)
        cv2.circle(current_frame, left_knee, point_radius, color, -1)
        cv2.circle(current_frame, right_knee, point_radius, color, -1)
        cv2.circle(current_frame, left_ankle, point_radius, color, -1)
        cv2.circle(current_frame, right_ankle, point_radius, color, -1)
        cv2.line(current_frame, left_hip, left_knee, color, line_width)
        cv2.line(current_frame, right_hip, right_knee, color, line_width)
        cv2.line(current_frame, left_hip, right_hip, color, line_width)
        cv2.line(current_frame, left_knee, left_ankle, color, line_width)
        cv2.line(current_frame, right_knee, right_ankle, color, line_width)

    
    # Save the skeleton image
    cv2.imwrite(video_path + f'skeletons_{i:05d}.jpg', current_frame)