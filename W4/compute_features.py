import numpy as np
import cv2
import os
import pandas as pd
import pickle
import tqdm
import matplotlib.pyplot as plt
# from tensorflow.keras import EfficientNetB0, Input, Model, GlobalAveragePooling2D

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


# # define a EfficientNetB0 model
# def create_model():
#     baseModel = EfficientNetB0(include_top=False, weights="imagenet")

#     outputs = GlobalAveragePooling2D(name="avg_pool")(baseModel.output)

#     model = Model(inputs=baseModel.input, outputs=outputs)

#     model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
#     return model



path_data = "data/"
# model = create_model()


    
sequences = ["S01"]

for seq in sequences:
    
    files = os.listdir(path_data + seq)
    files.sort()

    dict_frames = {}
    total_ids = 0
    added_ids = 0
    
    # perform a loop with tqdm to show the progress of the loop
    for file in files:
        # load the video
        frames = load_frames(path_data + seq + "/" + file + "/vdo.avi")

        # load the tracking data
        pred = pd.read_csv("tracking_data/" + file + "_track_eval.txt",header=None, sep=" ")

        #copy of pred
        pred_track = pred.copy()


        dict_frames[file] = {}
        for i in tqdm.tqdm(range(len(pred))):

            # get the frame number assuming that pred is a .txt file with the following format: frame_number, ID, x1, y1, x2, y2, confidence, -1, -1, -1
            frame_number = int(pred.iloc[i, 0])-1

            if frame_number not in dict_frames[file]:
                dict_frames[file][frame_number] = {}

            ID_og = int(pred.iloc[i, 1])

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

        

            # extract the histogram of the three channels separately
            hist_red = cv2.calcHist([cropped], [0], None, [256], [0, 256])
            hist_green = cv2.calcHist([cropped], [1], None, [256], [0, 256])
            hist_blue = cv2.calcHist([cropped], [2], None, [256], [0, 256])

            # concatenate the histograms horizontally
            hist = np.concatenate([hist_red, hist_green, hist_blue], axis=0)


            # normalize the histogram to have values between 0 and 1 without dividing by zero
            hist = hist / (hist.sum() + 1e-6)
    
            #hist = hist.tolist()
                # save the histogram in the dictionary at the total_frames position, in the total_ids position with the key as the string "hist"
            dict_frames[file][frame_number][ID_og + added_ids] = {"xtl": x1, "ytl": y1, "xbr": x2, "ybr": y2, "hist": hist}
            pred_track.iloc[i, 1] = ID_og + added_ids
            if ID_og + added_ids > total_ids:
                total_ids = ID_og + added_ids

        added_ids += total_ids


        # remove from memory the frames
        del frames

# save the dictionary in a json file
        pred_track.to_csv("tracking_data/" + file + "_track_eval_histrgb.txt", sep=" ", index=False, header=False)

        # save in a .pkl file
        with open("tracking_data/" + file + "_track_eval_histrgb.pkl", "wb") as f:
            pickle.dump(dict_frames, f)
            f.close()




            



