import cv2
import numpy as np
import time
from collections import deque
import mediapipe as mp



# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Load all images into a list
image_files = [f"pics/{i}.png" for i in range(1, 22)]  # Assuming you have images 1.png to 21.png
images = [cv2.imread(img) for img in image_files]

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize variables
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
last_displayed_index = None  # Track the last displayed image index
last_detection_time = time.time()
transition_in_progress = False  # To check if we're in the middle of a smoothing transition
start_index = None
end_index = None

# Store previous positions for smoothing
position_history = deque(maxlen=5)  # Keep track of the last 5 positions

while cap.isOpened():
    # Capture frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)

    # Convert the image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces using MediaPipe
    results = face_detection.process(rgb_frame)
    
    current_time = time.time()

    # Draw detections on the frame for debugging
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, w, h = (int(bboxC.xmin * frame_width), int(bboxC.ymin * frame_height),
                          int(bboxC.width * frame_width), int(bboxC.height * frame_height))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"X: {bboxC.xmin:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the webcam stream with detection annotations
    cv2.imshow("Webcam Stream", cv2.resize(frame, (320, 240)))  # Resize to smaller window

    # Handle the smoothing transition
    if transition_in_progress:
        if current_time - last_detection_time >= 0.05:
            # Perform the transition
            if start_index < end_index:
                next_index = start_index + 1
            else:
                next_index = start_index - 1

            # Display the next image in the transition
            cv2.imshow("Image Display", images[next_index - 1])  # Adjust index for 0-based list
            last_displayed_index = next_index
            
            # Update indices
            start_index = next_index
            last_detection_time = current_time  # Update the time

            # End the transition when we reach the target
            if start_index == end_index:
                transition_in_progress = False

    else:
        # Only update detection every 0.25 seconds and if no transition is in progress
        if current_time - last_detection_time >= 0.05:
            if results.detections:
                for detection in results.detections:
                    # Get the bounding box of the detected face
                    bboxC = detection.location_data.relative_bounding_box
                    center_x = bboxC.xmin + bboxC.width / 2

                    # Normalize the x-coordinate to the range [1, 21]
                    horizontal_position = (center_x * 20) + 1
                    position_history.append(horizontal_position)  # Add to position history

                    # Smooth the positions by taking the median of the last few positions
                    smoothed_position = np.median(position_history)
                    closest_index = round(smoothed_position)
                    closest_index = min(max(closest_index, 1), 21)  # Ensure index is within [1, 21]

                    # If the detected position corresponds to a different image, start smoothing transition
                    if closest_index != last_displayed_index:
                        start_index = last_displayed_index if last_displayed_index is not None else closest_index
                        end_index = closest_index
                        transition_in_progress = True
                        break

            last_detection_time = current_time  # Update the detection time

        # Continue displaying the last valid image
        if last_displayed_index is not None:
            cv2.imshow("Image Display", images[last_displayed_index - 1])  # Adjust index for 0-based list
            cv2.moveWindow("Image Display", 200, 200)  # Move "Image Display" window to (100, 100)
    # Wait for 'q' to quit
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
