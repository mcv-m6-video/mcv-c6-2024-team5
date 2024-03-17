import numpy as np
import cv2
import pickle
import pandas as pd

# load .pkl from tracking_data folder
with open("tracking_data/c002_track_eval_histrgb.pkl", "rb") as f:
    c002 = pickle.load(f)

with open("tracking_data/c003_track_eval_histrgb.pkl", "rb") as f:
    c003 = pickle.load(f)

# open a .txt file with the tracking data
pred_track_c003 = pd.read_csv("tracking_data/c003_track_eval_histrgb.txt", sep=" ", header=None)



# assume c003 is a nested dictionary with stucture {file: {frame: {id: {"xtl": x1, "ytl": y1, "xbr": x2, "ybr": y2, "hist": hist}}}}

row_txt = 0
# iterate over the dictionary c003 and compare the histograms with the ones in c002
for file_track in list(c003):
    if file_track not in list(c002):
        print("File: ", file_track, " is not present in the other camera")
        for frame_track in list(c003[file_track]):
            for id_track in list(c003[file_track][frame_track]):
                sim_list = []
                ids_list = []
                for file in list(c002):
                    for frame in list(c002[file]):
                        for id in list(c002[file][frame]):
                            # if the histograms are all nan values then the similarity index is set to inf
                            if np.isnan(c003[file_track][frame_track][id_track]["hist"]).all() or np.isnan(c002[file][frame][id]["hist"]).all():
                                sim = np.inf
                            
                            else:
                                sim = cv2.compareHist(np.array(c003[file_track][frame_track][id_track]["hist"]), np.array(c002[file][frame][id]["hist"]), cv2.HISTCMP_CHISQR)

                            sim_list.append(sim)
                            ids_list.append(id)
                if sim_list != []:
                    sim_list = np.array(sim_list)
                    ids_list = np.array(ids_list)   
                    max_sim = np.min(sim_list)
                    max_id = ids_list[np.argmin(sim_list)]
                    if max_sim <0.5:
                        # print("ID: ", id_track, " is the same as ID: ", max_id, " with similarity index: ", max_sim)
                        # change the ID in the c003 dataframe
                        pred_track_c003.loc[row_txt, 1] = max_id
                
                row_txt += 1
                #     else:
                #         print("ID: ", id_track, " is not present in the other camera")
                # else:
                #     print("ID: ", id_track, " is not present in the other camera")

# save the dataframe with the new IDs
pred_track_c003.to_csv("tracking_data/c003_track_eval_good_rgb.txt", sep=" ", index=False, header=False)