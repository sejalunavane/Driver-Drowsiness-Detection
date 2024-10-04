# Import necessary libraries
from scipy.spatial import distance  # For calculating the Euclidean distance between points
from imutils import face_utils  # Utility functions for facial landmarks
from pygame import mixer  # For playing alert sound
import imutils  # Image processing library
import dlib  # For facial detection and landmark prediction
import cv2  # OpenCV for real-time computer vision
import os  # For file operations (if needed)
import datetime  # To timestamp logs

# Initialize the pygame mixer for playing sounds
mixer.init()

# Load the sound file that will play during an alert
mixer.music.load("music.wav")

# Create a directory to store logs or outputs (if needed)
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Output directory created: {output_dir}")

# Log the start of the script
print(f"Script started at: {datetime.datetime.now()}")

# Function to log EAR values to a file (for analysis later)
def log_ear(ear_value, frame_count):
    """
    Log the eye aspect ratio (EAR) value for each frame to a file.
    
    Args:
    ear_value: A float representing the calculated EAR.
    frame_count: An integer representing the current frame number.
    """
    with open(os.path.join(output_dir, "ear_log.txt"), "a") as f:
        f.write(f"Frame {frame_count}: EAR={ear_value}\n")
    print(f"Logged EAR for frame {frame_count}")

# Function to calculate the Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    """
    Calculate the eye aspect ratio (EAR) for detecting drowsiness.

    The eye aspect ratio is computed using the vertical and horizontal distances
    between the eye landmarks and provides a single scalar value.
    
    Args:
    eye: List of (x, y)-coordinates representing the eye landmarks.
    
    Returns:
    ear: A float representing the eye aspect ratio.
    """
    # Calculate the vertical distances between two sets of eye landmarks
    A = distance.euclidean(eye[1], eye[5])  # Vertical distance 1
    B = distance.euclidean(eye[2], eye[4])  # Vertical distance 2
    
    # Calculate the horizontal distance between two eye landmarks
    C = distance.euclidean(eye[0], eye[3])  # Horizontal distance
    
    # Eye aspect ratio formula to calculate average vertical distance to horizontal distance
    ear = (A + B) / (2.0 * C)
    
    # Return the calculated EAR value
    return ear

# Threshold for EAR: If below this value, it is considered that the eyes are closed.
thresh = 0.25

# The number of consecutive frames the eyes must be closed before triggering an alert
frame_check = 20

# Initialize dlib's face detector (HOG-based) to detect faces in a frame
detect = dlib.get_frontal_face_detector()

# Initialize dlib's shape predictor for facial landmark detection
# This predictor is based on the 68 facial landmarks model
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Facial landmark indices for left and right eyes from the 68-point model
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Start capturing video from the webcam (camera index 0 by default)
cap = cv2.VideoCapture(0)

# Initialize a flag to count the consecutive frames where eyes are detected as closed
flag = 0

# Frame counter to log EAR values
frame_counter = 0

# Function to save a snapshot when the alert is triggered
def save_snapshot(frame, timestamp):
    """
    Save a snapshot of the current frame when an alert is triggered.
    
    Args:
    frame: The current video frame captured from the webcam.
    timestamp: The current timestamp used to name the file.
    """
    filename = os.path.join(output_dir, f"alert_snapshot_{timestamp}.png")
    cv2.imwrite(filename, frame)
    print(f"Snapshot saved: {filename}")

# Function to display status message (optional)
def display_status_message(message):
    """
    Display a status message in the console.
    
    Args:
    message: A string representing the message to be displayed.
    """
    print(f"STATUS: {message}")

# Begin the loop to continuously capture frames from the webcam
while True:
    # Capture a single frame from the webcam
    ret, frame = cap.read()

    # Resize the frame to a smaller width for faster processing (optional)
    frame = imutils.resize(frame, width=450)

    # Convert the frame from color (BGR) to grayscale for easier facial landmark detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use dlib's face detector to detect faces in the grayscale frame
    subjects = detect(gray, 0)

    # Loop over the detected faces in the current frame
    for subject in subjects:
        # Predict the facial landmarks for the detected face using the pre-trained model
        shape = predict(gray, subject)

        # Convert the facial landmarks into a NumPy array format (x, y)-coordinates
        shape = face_utils.shape_to_np(shape)

        # Extract the (x, y)-coordinates of the left and right eyes from the facial landmarks
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Calculate the Eye Aspect Ratio (EAR) for both the left and right eyes
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # Calculate the average EAR between both eyes to get a single EAR value
        ear = (leftEAR + rightEAR) / 2.0

        # Log the EAR value for this frame
        log_ear(ear, frame_counter)

        # Compute the convex hulls for the left and right eyes (for drawing eye contours)
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        # Draw contours around the left and right eyes on the frame using OpenCV
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # If the EAR is below the threshold, it indicates the eyes are closed
        if ear < thresh:
            # Increment the flag counter for each consecutive frame where the eyes are closed
            flag += 1

            # Print the current frame count where the eyes are closed for debugging
            print(flag)

            # If the eyes have been closed for a sufficient number of consecutive frames
            if flag >= frame_check:
                # Display an alert message on the frame to indicate drowsiness detection
                cv2.putText(frame, "*****ALERT!*****", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Display the same alert message at the bottom of the frame for emphasis
                cv2.putText(frame, "*****ALERT!*****", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Play the alert sound using the mixer module when drowsiness is detected
                mixer.music.play()

                # Save a snapshot of the frame with the alert
                save_snapshot(frame, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

                # Log the alert status
                display_status_message("ALERT Triggered! Drowsiness detected.")

        # If the EAR is above the threshold (eyes are open), reset the flag
        else:
            # Reset the flag to 0, indicating that the eyes are no longer closed
            flag = 0

    # Display the current frame in a window titled "Frame" using OpenCV
    cv2.imshow("Frame", frame)

    # Increment the frame counter
    frame_counter += 1

    # Check if the user pressed the 'q' key to quit the video stream
    key = cv2.waitKey(1) & 0xFF

    # If the 'q' key is pressed, break the loop and end the video capture
    if key == ord("q"):
        display_status_message("Video capture stopped by user.")
        break

# After the loop, release the video capture object and close all OpenCV windows
cv2.destroyAllWindows()

# Release the video capture object to free up the webcam
cap.release()

# Log the end of the script
print(f"Script ended at: {datetime.datetime.now()}")