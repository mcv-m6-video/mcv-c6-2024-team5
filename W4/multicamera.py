import numpy as np
import cv2
import os
import pandas as pd
import pickle
import tqdm
from skimage.feature import local_binary_pattern
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



            cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            
            #prediction = model.predict(np.expand_dims(cropped, axis=0))

            # compute LBP 
            lbp = local_binary_pattern(cropped_gray, 8, 1, method="uniform")

            import pdb; pdb.set_trace()


            # extract the histogram of the cropped image


            hist = (cv2.calcHist([cropped], [0], None, [256], [0, 256]))

            # normalize the histogram
            try:
                hist = hist / hist.sum()
            except:
                hist = np.zeros((256, 1))

            #hist = hist.tolist()

            # compare the histogram with the histograms of the previous frames
            if i == 0:
                # save the histogram in the dictionary at the total_frames position, in the total_ids position with the key as the string "hist"
                dict_frames[file][frame_number][ID_og + added_ids] = {"xtl": x1, "ytl": y1, "xbr": x2, "ybr": y2, "hist": hist}
                pred_track.iloc[i, 1] = ID_og + added_ids
                if ID_og + added_ids > total_ids:
                    total_ids = ID_og + added_ids

            else:
                tracked = False
            # compare the histogram with all the other histograms present in the dictionary
                sim_list = []
                ids_list = []
                for file_track in list(dict_frames):
                    if file_track!= file:
                        for frame_track in list(dict_frames[file_track]):
                            for id_track in list(dict_frames[file_track][frame_track]):
                            
                            # sim = 0.5 * np.sum(((np.array(hist) - np.array(dict_frames[file][frame][id]["hist"])) ** 2) / (np.array(hist) + np.array(dict_frames[file][frame][id]["hist"]) + 1e-10))
                                



                                sim = cv2.compareHist(np.array(hist), np.array(dict_frames[file_track][frame_track][id_track]["hist"]), cv2.HISTCMP_CHISQR)
                                print(sim)
                                if sim == np.nan:
                                    sim = np.inf
                                sim_list.append(sim)
                                ids_list.append(id_track)


                                

                # find the maximum similarity index
                if sim_list != []:
                    sim_list = np.array(sim_list)
                    ids_list = np.array(ids_list)   
                    max_sim = np.max(sim_list)
                    max_id = ids_list[np.argmax(sim_list)]
                    if max_sim <0.275:
                        tracked = True
                        tracked_ID = max_id
                
                            
                if tracked:
                    dict_frames[file][frame_number][tracked_ID] = {"xtl": x1, "ytl": y1, "xbr": x2, "ybr": y2, "hist": hist}
                    pred_track.iloc[i, 1] = tracked_ID
                else:
                    dict_frames[file][frame_number][ID_og+ added_ids] = {"xtl": x1, "ytl": y1, "xbr": x2, "ybr": y2, "hist": hist}
                    pred_track.iloc[i, 1] = ID_og + added_ids
                    if ID_og + added_ids > total_ids:
                        total_ids = ID_og + added_ids
        added_ids += total_ids


        # remove from memory the frames
        del frames

# save the dictionary in a json file
        pred_track.to_csv("tracking_data/" + file + "_track_eval_hist.txt", sep=" ", index=False, header=False)

        # save in a .pkl file
        with open("tracking_data/" + file + "_track_eval_hist.pkl", "wb") as f:
            pickle.dump(dict_frames, f)
            f.close()




            



