import cv2
import numpy as np
import source.global_variables as gv

## VISUALIZATION FUNCTIONS

def show_binary_frames(binary_frames, total_frames, gt, preds, aps, map):
    for i in range(int(total_frames * gv.Params.FRAMES_PERCENTAGE), total_frames):
        binary_frame = binary_frames[i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)]
        # Change to BGR for visualization
        binary_frame = cv2.cvtColor(binary_frame, cv2.COLOR_GRAY2BGR)
        binary_frame = add_rectangles_to_frame(binary_frame, gt[i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)], (0, 0, 255))
        binary_frame = add_rectangles_to_frame(binary_frame, preds[i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)], (0, 255, 0))
        put_text_top_left(binary_frame, f"AP: {aps[i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)]:.5f}, Frame: {i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)}. mAP of full video: {map:.5f}")
        cv2.imshow('binary_frame', binary_frame)
        cv2.waitKey(0)

def add_rectangles_to_frame(frame, boxes, color):
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    return frame

def put_text_top_left(frame, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2
    cv2.putText(frame, text, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

def show_frame_with_pred(cap, binary_frames, total_frames, gt, preds, aps, map):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * gv.Params.FRAMES_PERCENTAGE))
    for i in range(int(total_frames * gv.Params.FRAMES_PERCENTAGE), total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #frame = frames[i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)]
        binary_frame = binary_frames[i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)]

        # combine binary_frame as an overlay of frame with the binarization in pink color
        binary_frame_color = np.zeros(frame.shape, dtype=np.uint8)
        
        # fill the binary_frame_color with the binary_frame items that were not zeros as pink
        binary_frame_color[binary_frame != 0] = [255, 0, 255]
        overlay = cv2.addWeighted(frame, 0.7, binary_frame_color, 0.3, 0)
        overlay = add_rectangles_to_frame(overlay, gt[i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)], (0, 0, 255))
        overlay = add_rectangles_to_frame(overlay, preds[i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)], (0, 255, 0))
        put_text_top_left(overlay, f"AP: {aps[i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)]:.5f}, Frame: {i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)}. mAP of full video: {map:.5f}")
        
        if i == gv.Params.FRAME_TO_ANALYZE:
            # Save the overlay image for the selected frame
            cv2.imwrite(f"{gv.Params.PATH_RUN}overlay_{i}.png", overlay)

        cv2.imshow('overlay', overlay)
        cv2.waitKey(0)
    
## VISUALIZATION FUNCTIONS W2
        
def display_frame_with_overlay(frame, gt_boxes, pred_boxes, ap, map, display):
    frame = add_rectangles_to_frame(frame, gt_boxes, (0, 255, 0)) # Green for the GT
    frame = add_rectangles_to_frame(frame, pred_boxes, (0, 0, 255)) # Red for the predictions
    put_text_top_left(frame, f"AP: {ap:.5f}; mAP for the full video: {map:.5f}")
    if display:
        cv2.imshow('overlay', frame)
        cv2.waitKey(1)
    return frame