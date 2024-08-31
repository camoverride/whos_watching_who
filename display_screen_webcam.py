import os
import cv2
import numpy as np
import time
from collections import deque
import yaml
import mediapipe as mp
from threading import Thread


# Load the YAML configuration file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Read the images (cropped portion) into memory
num_locations = len(os.listdir(config["path_to_image_directory"]))
cropped_height = config["height"] - 960 - 905
cropped_width = config["width"] - 395 - 395
images = np.memmap(config["path_to_image_memmap"],
                   dtype=np.uint8,
                   mode='r',
                   shape=(num_locations, cropped_height, cropped_width, 3))

# Load a background image (you can modify the path to your background image)
background_image = cv2.imread(config["path_to_background_image"])

# Rotate screen
os.environ["DISPLAY"] = ':0'
os.system("WAYLAND_DISPLAY=wayland-1 wlr-randr --output HDMI-A-2 --transform 90")

# Hide the mouse
os.system("unclutter -idle 0 &")

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize variables
frame_width = config["width"]
frame_height = config["height"]
last_displayed_index = None
transition_in_progress = False
start_index = None
end_index = None

# Store previous positions for smoothing
position_history = deque(maxlen=5)  # Keep track of the last 5 positions

# Create a fullscreen window for image display
cv2.namedWindow("Image Display", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Image Display", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Pre-read the initial image from memmap
initial_overlay = images[num_locations // 2]
background_image[960:960+cropped_height, 395:395+cropped_width] = initial_overlay
cv2.imshow("Image Display", background_image)

# Function for handling face detection in a separate thread
def detect_faces():
    global face_detection_results, frame_ready, detection_thread_active
    while detection_thread_active:
        ret, frame = cap.read()
        if ret:
            # Flip the frame horizontally, to mimic a mirror.
            frame = cv2.flip(frame, 1)
            face_detection_results = face_detection.process(frame)
            frame_ready = True

# Variables for threading
face_detection_results = None
frame_ready = False
detection_thread_active = True

# Start the face detection thread
detection_thread = Thread(target=detect_faces)
detection_thread.start()

# Function to update the display
def update_display(index):
    overlay_image = images[index - 1]  # Read only the needed part of memmap
    background_image[960:960+cropped_height, 395:395+cropped_width] = overlay_image
    cv2.imshow("Image Display", background_image)

# Main event loop
while True:
    current_time = time.time()

    if transition_in_progress:
        # Speed up transition handling
        next_index = min(end_index, start_index + config["stride"]) if start_index < end_index else max(end_index, start_index - config["stride"])
        update_display(next_index)
        last_displayed_index = next_index
        start_index = next_index

        # End the transition when we reach the target
        if start_index == end_index:
            transition_in_progress = False

    else:
        if frame_ready:
            if face_detection_results and face_detection_results.detections:
                for detection in face_detection_results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    center_x = bboxC.xmin + bboxC.width / 2
                    horizontal_position = (center_x * (num_locations - 1)) + 1
                    position_history.append(horizontal_position)

                    # Smooth the positions by taking the mean of the last few positions
                    smoothed_position = np.mean(position_history) if len(position_history) > 1 else position_history[0]
                    closest_index = round(smoothed_position)
                    closest_index = min(max(closest_index, 1), num_locations)

                    if closest_index != last_displayed_index:
                        start_index = last_displayed_index if last_displayed_index is not None else closest_index
                        end_index = closest_index
                        transition_in_progress = True
                        break

            frame_ready = False

        if last_displayed_index is not None and not transition_in_progress:
            update_display(last_displayed_index)
    
    # Wait for 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Stop the detection thread
detection_thread_active = False
detection_thread.join()

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
