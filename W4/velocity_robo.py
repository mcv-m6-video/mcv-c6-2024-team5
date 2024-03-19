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

            # Draw speed on the frame at the top-right of the bounding box 
            cv2.putText(frame, f"Speed: {speed:.2f} km/h", (int(x2), int(y2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Update speed information for the car ID
            speeds[tracker_id]["total_distance"] += distance
            speeds[tracker_id]["total_time"] += time

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

model_path = "./S03.pt"
print(f"Loading YOLO model from: {model_path}")

# Load the YOLO model
model = YOLO(model_path)
print("YOLO model loaded successfully.")

# Initialize ByteTrack tracker
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Video file path
video_path = "data/aic19-track1-mtmc-train/train/S03/C010/vdo.avi"
print(f"Reading video from: {video_path}")

# Initialize video capture
video_capture = cv2.VideoCapture(video_path)

# Get video properties
fps = video_capture.get(cv2.CAP_PROP_FPS)
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('fps: ', fps, 'width: ', width, 'height: ', height)

# Initialize the ViewTransformer
view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

# Create video writer for output
output_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Video info for speed calculation
video_info = sv.VideoInfo.from_video_path(video_path)
coordinates = defaultdict(lambda: deque(maxlen=int(video_info.fps)))

# Dictionary to store speed information for each car ID
speeds = defaultdict(lambda: {"total_distance": 0, "total_time": 0})

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    # Draw source points on the frame for visualization
    for point in SOURCE:
        cv2.circle(frame, tuple(map(int, point)), 5, (0, 255, 0), -1)

    # Process the frame using the callback function
    annotated_frame = callback(frame, 0)

    # Check if the frame size is valid
    if annotated_frame.shape[0] > 0 and annotated_frame.shape[1] > 0:
        
        # Resize the window to fit the image dimensions
        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Frame', width, height)

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

# Release video capture and video writer
video_capture.release()
out.release()
cv2.destroyAllWindows()
