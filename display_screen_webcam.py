import os
import cv2
import numpy as np
import time
from collections import deque
import yaml
import mediapipe as mp


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
last_displayed_index = None  # Track the last displayed image index
last_detection_time = time.time()
transition_in_progress = False  # To check if we're in the middle of a smoothing transition
start_index = None
end_index = None

# Store previous positions for smoothing
position_history = deque(maxlen=5)  # Keep track of the last 5 positions

# Create a fullscreen window for image display
cv2.namedWindow("Image Display", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Image Display", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Display the initial background with the central image overlaid
initial_overlay = images[num_locations // 2]
background_image[960:960+cropped_height, 395:395+cropped_width] = initial_overlay
cv2.imshow("Image Display", background_image)

# Main event loop
while True:
    # Capture frame from webcam
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture image from webcam.")
        break

    # Flip the frame horizontally, to mimic a mirror.
    frame = cv2.flip(frame, 1)

    # Detect faces using MediaPipe on the flipped frame
    results = face_detection.process(frame)

    # Draw detections on the frame for debugging
    if config["debug"] and results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, w, h = (int(bboxC.xmin * frame.shape[1]), 
                        int(bboxC.ymin * frame.shape[0]),
                        int(bboxC.width * frame.shape[1]), 
                        int(bboxC.height * frame.shape[0]))

            # Adjust the x-coordinate for the mirrored frame correctly
            x = frame.shape[1] - (x + w)  # This correctly mirrors the x-coordinate

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"X: {bboxC.xmin:.2f}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the webcam stream with detection annotations
        debug_window_name = "Webcam Stream"
        cv2.namedWindow(debug_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(debug_window_name, 320, 240)  # Set size of the debug window
        cv2.moveWindow(debug_window_name, 0, frame_height - 240)  # Move window to bottom-left
        cv2.imshow(debug_window_name, frame)  # Show frame in the debug window

    current_time = time.time()

    # Handle the smoothing transition
    if transition_in_progress:
        if current_time - last_detection_time >= 0.00005:
            # Perform the transition
            if start_index < end_index:
                next_index = min(end_index, start_index + config["stride"])
            else:
                next_index = max(end_index, start_index - config["stride"])

            # Overlay the next image in the transition onto the background
            background_image[960:960+cropped_height, 395:395+cropped_width] = images[next_index - 1]
            cv2.imshow("Image Display", background_image)
            last_displayed_index = next_index
            
            # Update indices
            start_index = next_index
            last_detection_time = current_time  # Update the time

            # End the transition when we reach the target
            if start_index == end_index:
                transition_in_progress = False

    else:
        # Only update detection every 0.05 seconds and if no transition is in progress
        if current_time - last_detection_time >= 0.0005:
            if results.detections:
                for detection in results.detections:
                    # Get the bounding box of the detected face
                    bboxC = detection.location_data.relative_bounding_box
                    center_x = bboxC.xmin + bboxC.width / 2

                    # Normalize the x-coordinate to the range [1, NUM_LOCATIONS]
                    horizontal_position = (center_x * (num_locations - 1)) + 1
                    position_history.append(horizontal_position)  # Add to position history

                    # Smooth the positions by taking the median of the last few positions
                    smoothed_position = np.median(position_history)
                    closest_index = round(smoothed_position)
                    closest_index = min(max(closest_index, 1), num_locations)  # Ensure index is within [1, NUM_LOCATIONS]

                    # If the detected position corresponds to a different image, start smoothing transition
                    if closest_index != last_displayed_index:
                        start_index = last_displayed_index if last_displayed_index is not None else closest_index
                        end_index = closest_index
                        transition_in_progress = True
                        break

            last_detection_time = current_time  # Update the detection time

        # Continue displaying the last valid image
        if last_displayed_index is not None:
            background_image[960:960+cropped_height, 395:395+cropped_width] = images[last_displayed_index - 1]
            cv2.imshow("Image Display", background_image)
    
    # Wait for 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
