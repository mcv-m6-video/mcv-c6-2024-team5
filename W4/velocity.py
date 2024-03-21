import cv2
import numpy as np
from collections import defaultdict, deque
import supervision as sv
from ultralytics import YOLO

# Ref: https://supervision.roboflow.com/how_to/track_objects/?ref=blog.roboflow.com
# Ref: https://blog.roboflow.com/estimate-speed-computer-vision/

# Function for the tracking with the detections from YOLO and to calculate the speed
def callback(frame: np.ndarray, _: int) -> np.ndarray:
    # Use the YOLO model to get detections on the frame
    if CUSTOM:
        results = model.predict(frame,classes=[2])[0]
    else:
        results = model(frame)[0]
    # Convert the YOLO detections to a custom 'Detections' object
    detections = sv.Detections.from_ultralytics(results)
    # Update the tracker with the new detections
    detections = tracker.update_with_detections(detections)

    labels = [
        f"#{tracker_id} {results.names[class_id]}"
        for class_id, tracker_id
        in zip(detections.class_id, detections.tracker_id)
    ]

    # Calculate speed and draw bounding boxes
    for tracker_id, bbox in zip(detections.tracker_id, detections.xyxy):
        x1, y1, x2, y2 = bbox[:4]
        x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2
        # Transform the center of the bounding box using the view transformer
        transformed_point = view_transformer.transform_points(np.array([[x_center, y_center]]))[0]

        # Update detections with transformed bounding boxes
        bbox[:4] = [x_center - (x2 - x1) / 2, y_center - (y2 - y1) / 2,
                    x_center + (x2 - x1) / 2, y_center + (y2 - y1) / 2]

        # Store the transformed coordinates for speed calculation
        coordinates[tracker_id].append(transformed_point[1])

        # Calculate speed
        if len(coordinates[tracker_id]) > video_info.fps / 2:
            coordinate_start = coordinates[tracker_id][-1]
            coordinate_end = coordinates[tracker_id][0]
            distance = abs(coordinate_start - coordinate_end)
            time = len(coordinates[tracker_id]) / video_info.fps
            speed = distance / time * 3.6  # Convert to km/h

            speed_display = int(speed)  # Convert speed to integer for display

            # If speed is over 50/80 km/h, draw bounding box in RED
            speed_limit = 50
            if speed_display > speed_limit: 
                cv2.putText(frame, f"Speed: {speed_display} km/h", (int(x2), int(y2)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, f"Speed: {speed_display} km/h", (int(x2), int(y2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # Update speed information for the car ID
            speeds[tracker_id]["speeds"].append(speed)
            speeds[tracker_id]["total_distance"] += distance
            speeds[tracker_id]["total_time"] += time

            # Append the speed to a text file for each frame
            speed_data_file.write(f"Frame: {frame_count} | Car #{tracker_id} Speed: {speed} km/h\n")

    # Annotate the frame with boxes and labels
    annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)

    return annotated_frame

# Class for perspective transformation
class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        # Convert points to 2D array if not already
        if points.ndim == 1:
            points = np.array([points])

        # Add a column of ones to the points for homogeneous coordinates
        ones_column = np.ones((points.shape[0], 1), dtype=np.float32)
        points_homogeneous = np.hstack((points, ones_column))
        # Apply the transformation matrix
        transformed_points_homogeneous = np.dot(points_homogeneous, self.m.T)
        # Convert back to Cartesian coordinates
        transformed_points = transformed_points_homogeneous[:, :2] / transformed_points_homogeneous[:, 2:]

        return transformed_points

CUSTOM = True

if CUSTOM:
    # For daylight video
    # SOURCE = np.array([
    #     [613.33, 555.94],  # Top-left corner of the road
    #     [1212.20, 555.94],  # Top-right corner of the road
    #     [201.60, 1022.87],  # Bottom-left corner of the road
    #     [1328.00, 1022.87] # Bottom-right corner of the road
    # ])   
    
    # For nightlight video
    SOURCE = np.array([
        [654.10, 435.50],  # Top-left corner of the road
        [1231.70, 435.50],  # Top-right corner of the road
        [279.70, 890.09],  # Bottom-left corner of the road
        [1353.10, 890.09]    # Bottom-right corner of the road
    ])

    TARGET = np.array([
        [  0,   0],
         [8,  0],
         [  0,   19],
         [8,   19]
        ])
else:
    # Define source points for perspective transformation (road rectangle)
    SOURCE = np.array([
        [538.50, 165.64],  # Top-left corner of the road
        [1114.12, 165.64],  # Top-right corner of the road
        [317.25, 541.70],  # Bottom-left corner of the road
        [1588.52, 541.70]    # Bottom-right corner of the road
    ])

    # Define target points 
    # Assuming 10m wide and 80m long
    TARGET = np.array([
        [[  0,   0],
        [9,   0],
        [  0,   79],
        [9,   79]]
    ])

model_path = "S03.pt"
# Load the YOLO model
# model = YOLO(model_path)

if CUSTOM:
    # pretrained model from YOLO to detecth only cars
    model = YOLO("yolov5su.pt")
    model.classes = [2]  # Only detect cars
else: 
    model = YOLO(model_path)
print("YOLO model loaded successfully.")

# Initialize ByteTrack tracker
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=2)

# Video file path
if CUSTOM:
    video_path = "data/aic19-track1-mtmc-train/train/night_cut.mp4"
else:
    video_path = "data/aic19-track1-mtmc-train/train/S03/C010/vdo.avi"
print(f"Reading video from: {video_path}")

# Initialize video capture
video_capture = cv2.VideoCapture(video_path)

# Get video properties
fps = video_capture.get(cv2.CAP_PROP_FPS)
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
# print('fps: ', fps, 'width: ', width, 'height: ', height)

# Initialize the ViewTransformer
view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

# Create video writer for output
if CUSTOM:
    output_path = "output_video_ronda_nit.mp4"
else:
    output_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Video info for speed calculation
video_info = sv.VideoInfo.from_video_path(video_path)
coordinates = defaultdict(lambda: deque(maxlen=int(video_info.fps)))

# Dictionary to store speed information for each car ID
speeds = defaultdict(lambda: {"speeds": [], "total_distance": 0, "total_time": 0})

# Open a text file to save speed data
speed_data_file = open("speed_data.txt", "w")
frame_count = 0  # Counter for frame number

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    frame_count += 1  # Increment frame count
    # Draw source points on the frame for visualization
    for point in SOURCE:
        cv2.circle(frame, tuple(map(int, point)), 5, (0, 255, 0), -1)

    # Process the frame using the callback function
    annotated_frame = callback(frame, 0)

    # Check if the frame size is valid
    if annotated_frame.shape[0] > 0 and annotated_frame.shape[1] > 0:
        
        # Resize the window to fit the image dimensions
        # cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Frame', width, height)

        # Write the frame to the output video
        out.write(annotated_frame)

        # Display the frame
        cv2.imshow('Frame', annotated_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Calculate and print average speed for each car ID
print("Average Speeds:")
for tracker_id, data in speeds.items():
    total_distance = data["total_distance"]
    total_time = data["total_time"]
    if total_time > 0:
        average_speed = (total_distance / total_time) * 3.6  # Convert to km/h
        print(f"Car #{tracker_id}: {average_speed:.2f} km/h")

# Calculate and print median speed for each car ID
print("Median Speeds:")
for tracker_id, data in speeds.items():
    # Extract the speeds list for this car
    car_speeds = data["speeds"]

    # Calculate the median speed
    if car_speeds:
        median_speed = np.median(car_speeds)
        num_speeds = len(car_speeds)  # Number of speeds recorded
        if num_speeds > 10: # avoid some false positives
            print(f"Car #{tracker_id}: Median Speed: {median_speed:.2f} km/h")
    else:
        print(f"Car #{tracker_id}: No speed data available")

# Write speed data to the text file
speed_data_file.close()

# Release video capture and video writer
video_capture.release()
out.release()
cv2.destroyAllWindows()
