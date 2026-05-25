# Starter Code for Gesture Detection using MediaPipe and OpenCV from Google MediaPipe's documentation: https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer/python
# Youtube Video's tutorials and etc.
# looking forward to learning and implementing something unique as I dive into a bit of machine learning and AI in this GITHUB repository.
# IMPORTS
import time
import mediapipe as mp 
from mediapipe.tasks import python 
import cv2 
from pathlib import Path 
from typing import List, Tuple, Optional

# MODEL PATH
current_dir = Path(__file__).parent
model_path = current_dir / "gesture_recognizer.task"

BaseOptions = mp.tasks.BaseOptions
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
VisionRunningMode = mp.tasks.vision.RunningMode
Image = mp.Image


_latest_gesture: Optional[str] = None

_latest_landmarks_norm: Optional[List[Tuple[float, float, float]]] = None


HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4) #THUMB
    (0,5), (5,6), (6,7), (7,8) #INDEX
    (0,9), (9,10), (10,11), (11,12) #MIDDLE
    (0,13), (13,14), (14,15), (15,16) #RING
    (0,17), (17,18), (18,19), (19,20) #PINKY
    
]

# Function to draw landmarks and connections on the persons hand
def print_result(result, output_image, timestamp_ms: int):
    """ Callback function to print results from the gesture recognizer. """
    



# Running the Gesture Recognizer in video mode with OpenCV 
    def main():
        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.VIDEO,
            result_callback=print_result)
        
        
        # Create the object 
        with GestureRecognizer.create_from_options(options) as recognizer:



        # Open the webcam and start processing video frames
        # Validation
            cam = cv2.VideoCapture(0)
        if not cam.isOpened():
           print("Error: Could not open webcam.")
           return 
    


