import numpy as np
import cv2
import os
import pandas as pd

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



    # perform a loop with tqdm to show the progress of the loop
    for file in files:
        if file == "c002":
            continue
        # load the video
        frames = load_frames(path_data + seq + "/" + file + "/vdo.avi")

        # load the tracking data
        if file == "c002":
            pred = pd.read_csv("tracking_data/c002_track_eval.txt",header=None, sep=" ")
        elif file == "c003":              
            pred = pd.read_csv("tracking_data/c003_track_eval_good_rgb.txt",header=None, sep=" ")

        

        # draw the bounding boxes with the id in each frame according to the tracking data
        for i in range(len(pred)):
            frame_number = int(pred.iloc[i, 0])-1
            ID = int(pred.iloc[i, 1])

            x,y,w,h = pred.iloc[i, 2:6].values
            xtl = x
            ytl = y
            xbr = x+w
            ybr = y+h
            cv2.rectangle(frames[frame_number],(int(xtl),int(ytl)),(int(xbr),int(ybr)),(0,255,0),2)
            cv2.putText(frames[frame_number], str(ID), (int(xtl),int(ytl)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

        # visualize the frames with the bounding boxes and the id
        for i in range(len(frames)):
            cv2.imshow('Frame', frames[i])
            cv2.waitKey(0)

        # # save the frames with the bounding boxes and the id as a video
        # out = cv2.VideoWriter('output ' + file + '.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, (frames[0].shape[1],frames[0].shape[0]))
        # for i in range(len(frames)):
        #     out.write(frames[i])
        
        # out.release()
        # cv2.destroyAllWindows()

        del frames


