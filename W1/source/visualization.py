import cv2
import source.global_variables as gv

## VISUALIZATION FUNCTIONS

def show_binary_frames(binary_frames, frame_dict, total_frames):
    for i in range(int(total_frames * gv.Params.FRAMES_PERCENTAGE), total_frames):
        binary_frame = binary_frames[i - int(total_frames * gv.Params.FRAMES_PERCENTAGE)]
        binary_frame_to_viz = add_rectangles_to_frame(binary_frame, frame_dict, i)
        cv2.imshow('binary_frame', binary_frame_to_viz)
        cv2.waitKey(0)

def add_rectangles_to_frame(frame, frame_dict, frame_number):
    for car_id, car in frame_dict[str(frame_number)].items():
        if car['is_parked']:
            continue
        x1, y1, x2, y2 = int(car['xtl']), int(car['ytl']), int(car['xbr']), int(car['ybr'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return frame