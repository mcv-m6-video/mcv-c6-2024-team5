import cv2
import source.global_variables as gv

## VISUALIZATION FUNCTIONS

def show_binary_frames(binary_frames, total_frames, gt, preds, aps):
    for i in range(int(total_frames * gv.Params.FRAMES_PERCENTAGE), total_frames):
        binary_frame = binary_frames[i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)]
        # Change to BGR for visualization
        binary_frame = cv2.cvtColor(binary_frame, cv2.COLOR_GRAY2BGR)
        binary_frame = add_rectangles_to_frame(binary_frame, gt[i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)], (0, 0, 255))
        binary_frame = add_rectangles_to_frame(binary_frame, preds[i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)], (0, 255, 0))
        put_text_top_left(binary_frame, f"AP: {aps[i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)]:.5f}, Frame: {i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)}")
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