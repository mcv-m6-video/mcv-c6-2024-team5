import cv2
import numpy as np


class VideoCaptureWrapper:
    def __init__(self, video_sources, caps_to_show=6):
        self.video_captures = [cv2.VideoCapture(video_source) for video_source in video_sources]
        self.num_captures = len(self.video_captures)
        self.frames = []
        self.window_name = "Sequence"
        self.caps_to_show = caps_to_show

        self.window_width = 1920
        self.window_height = 1080
        # Calculate the number of rows and columns depending on the number of video sources
        if self.caps_to_show == 1:
            self.rows = 1
            self.cols = 1
        elif self.caps_to_show == 2:
            self.rows = 1
            self.cols = 2
        elif self.caps_to_show == 3 or self.caps_to_show == 4:
            self.rows = 2
            self.cols = 2
        elif self.caps_to_show == 5 or self.caps_to_show == 6:
            self.rows = 2
            self.cols = 3
        elif self.caps_to_show == 7 or self.caps_to_show == 8:
            self.rows = 2
            self.cols = 4
        elif self.caps_to_show == 9 or self.caps_to_show == 10:
            self.rows = 3
            self.cols = 4
        elif self.caps_to_show == 11 or self.caps_to_show == 12:
            self.rows = 3
            self.cols = 4
        elif self.caps_to_show == 13 or self.caps_to_show == 14:
            self.rows = 3
            self.cols = 5
        elif self.caps_to_show == 15 or self.caps_to_show == 16:
            self.rows = 4
            self.cols = 4
        else:
            self.rows = 5
            self.cols = 5

        # Calculate the width and height of each frame
        self.frame_width = int(self.window_width / self.cols)
        self.frame_height = int(self.window_height / self.rows)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_width, self.window_height)

        # Create object to save the video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter('output.avi', fourcc, 20.0, (self.window_width, self.window_height))

    def __iter__(self):
        return self

    def __next__(self):
        self.frames = []
        for cap in self.video_captures:
            ret, frame = cap.read()
            if not ret:
                cap.release()
                self.num_captures -= 1
                if self.num_captures == 0:
                    raise StopIteration
            self.frames.append(frame)
        return self.frames

    def collage(self, results):
        """Function to display the frames in a collage. It will resize the frames to fit the collage"""
        resized_frames = []
        for frame, bboxes in zip(self.frames, results):
            for bbox in bboxes:
                frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                # Write the ID of the tracked bounding box (bbox[4]) in the top left corner of the bounding box
                cv2.putText(frame, str(int(bbox[5])), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            resized_frames.append(cv2.resize(frame, (self.frame_width, self.frame_height)))
        # Create the collage as a black image of the size of the window
        collage = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
        # Fill the collage with the frames
        for i in range(self.rows):
            for j in range(self.cols):
                if len(resized_frames) == 0 or (i * self.cols + j) == self.caps_to_show:
                    break
                collage[i * self.frame_height:(i + 1) * self.frame_height, j * self.frame_width:(j + 1) * self.frame_width] = resized_frames.pop(0)
        # Display the collage
        cv2.imshow(self.window_name, collage)
        cv2.waitKey(1)
        # Save as video
        self.out.write(collage)




