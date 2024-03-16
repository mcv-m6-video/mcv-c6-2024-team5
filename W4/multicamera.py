import numpy as np
import cv2
import os
import pandas as pd
import json
import tqdm

def load_frames(path_to_video):
    cap = cv2.VideoCapture(path_to_video)
    if not cap.isOpened():
        raise ValueError("Error opening video file")
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

path_data = "data/"


    
sequences = ["S01"]

for seq in sequences:
    
    files = os.listdir(path_data + seq)
    files.sort()

    dict_frames = {}
    total_ids = 0

    # perform a loop with tqdm to show the progress of the loop
    for file in files:
        # load the video
        frames = load_frames(path_data + seq + "/" + file + "/vdo.avi")

        # load the tracking data
        pred = pd.read_csv("tracking_data/" + file + "_track_eval.txt",header=None, sep=" ")


        dict_frames[file] = {}
        for i in tqdm.tqdm(range(len(pred))):

            # get the frame number assuming that pred is a .txt file with the following format: frame_number, ID, x1, y1, x2, y2, confidence, -1, -1, -1
            frame_number = int(pred.iloc[i, 0])-1

            if frame_number not in dict_frames[file]:
                dict_frames[file][frame_number] = {}

            # get the bounding box
            bbox = pred.iloc[i, 2:6].values
            # bbox is in the format left,top,width,height
            # we need to convert it to left,right,top,bottom
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[0] + bbox[2])
            y2 = int(bbox[1] + bbox[3])

            # crop the bounding box from the frame
            cropped = frames[frame_number][y1:y2, x1:x2]

            # extract the histogram of the cropped image
            hist = cv2.calcHist([cropped], [0], None, [256], [0, 256])

            # normalize the histogram
            try:
                hist = hist / hist.sum()
            except:
                hist = np.zeros((256, 1))

            # compare the histogram with the histograms of the previous frames
            if i == 0:
                # save the histogram in the dictionary at the total_frames position, in the total_ids position with the key as the string "hist"
                dict_frames[file][frame_number][total_ids] = {"hist": hist, "bbox": bbox}
                total_ids += 1

            else:
            # compare the histogram with all the other histograms present in the dictionary
                for frame in list(dict_frames[file]):
                    for id in list(dict_frames[file][frame]):
                        # compare the histogram with the histogram of the id
                        sim = cv2.compareHist(hist, dict_frames[file][frame][id]["hist"], cv2.HISTCMP_CORREL)
                        if sim > 0.9:
                            # if they are similar save the histogram in the dictionary at the total_frames position, in the id position with the key as the string "hist"
                            dict_frames[file][frame_number][id] = {"hist": hist, "bbox": bbox}

                        else:
                            # if they are not similar create a new instance of the dictionary with a new id
                            dict_frames[file][frame_number][total_ids] = {"hist": hist, "bbox": bbox}
                            total_ids += 1
        # remove from memory the frames
        del frames

# save the dictionary in a json file

        with open('data.json', 'w') as fp:
            json.dump(dict_frames, fp)

        import pdb; pdb.set_trace()



            



