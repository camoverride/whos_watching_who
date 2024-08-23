import os
import cv2
import numpy as np
import time
from collections import deque
import yaml
import mediapipe as mp
from picamera2 import Picamera2



# Load the YAML configuration file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Read the images
num_locations = len(os.listdir(config["path_to_image_directory"]))
images = np.memmap(config["path_to_image_memmap"],
                   dtype=np.uint8,
                   mode='r',
                   shape=(num_locations, config["height"], config["width"], 3))

# Rotate screen
os.environ["DISPLAY"] = ':0'
os.system("WAYLAND_DISPLAY=wayland-1 wlr-randr --output HDMI-A-1 --transform 90")

# Hide the mouse
os.system("unclutter -idle 0 &")

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Initialize the PiCamera2 module
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (1280, 720), "format": "RGB888"}))
picam2.start()

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

# Show the initial image (center facing)
start_im = images[num_locations // 2]
cv2.imshow("Image Display", start_im)


# Main event loop.
while True:
    # Capture frame
    frame = np.array(picam2.capture_array(), dtype=np.uint8)

    # Convert the image to RGB
    rgb_frame = frame  # Already in RGB format
    
    # Detect faces using MediaPipe
    results = face_detection.process(rgb_frame)
    
    current_time = time.time()

    # Draw detections on the frame for debugging
    if config["debug"]:
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x, y, w, h = (int(bboxC.xmin * frame_width), int(bboxC.ymin * frame_height),
                            int(bboxC.width * frame_width), int(bboxC.height * frame_height))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"X: {bboxC.xmin:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the webcam stream with detection annotations
        debug_window_name = "Webcam Stream"
        cv2.namedWindow(debug_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(debug_window_name, 320, 240)  # Set size of the debug window
        cv2.moveWindow(debug_window_name, 0, frame_height - 240)  # Move window to bottom-left
        cv2.imshow(debug_window_name, frame)  # Show frame in the debug window

    # Handle the smoothing transition
    if transition_in_progress:
        if current_time - last_detection_time >= 0.0001:
            # Perform the transition
            if start_index < end_index:
                next_index = start_index + 1
            else:
                next_index = start_index - 1

            # Display the next image in the transition
            if images[next_index - 1] is not None:
                cv2.imshow("Image Display", images[next_index - 1])  # Adjust index for 0-based list
                last_displayed_index = next_index
            
            # Update indices
            start_index = next_index
            last_detection_time = current_time  # Update the time

            # End the transition when we reach the target
            if start_index == end_index:
                transition_in_progress = False

    else:
        # Only update detection every 0.05 seconds and if no transition is in progress
        if current_time - last_detection_time >= 0.05:
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
            if images[last_displayed_index - 1] is not None:
                cv2.imshow("Image Display", images[last_displayed_index - 1])  # Adjust index for 0-based list
    
    # Wait for 'q' to quit
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

# Release the capture and close windows
picam2.stop()
cv2.destroyAllWindows()
