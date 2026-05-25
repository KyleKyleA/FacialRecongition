# Starter Code for Gesture Detection using MediaPipe and OpenCV from Google MediaPipe's documentation: https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer/python
# IMPORTS
import mediapipe as mp 
from mediapipe.tasks import python 
from mediapipe.tasks.python import vision 
import cv2

# MODEL PATH
model_path = './gesture_recognizer.task'

base_options = BaseOptions(model_asset_path=model_path)

BaseOptions = mp.tasks.BaseOptions
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# CREATE THE GESTURE RECOGNIZER
def print_results(result: GestureRecognizerResult, output_image : mp.image, timestamp_ms: int):
    print('gesture recognition results: {}'.format(result.gestures))
    

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='/path/to/model.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_results)
with GestureRecognizer.create_from_options(options) as recognizer:
    # The detector is initialized. Use it here.
    # ...
    
    # Preparing Data Below
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
    else:
        print("Success: WebCam opened.")
    
    
    # convert the frame recieved from OpenCV to a MediaPipe's Image object.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)
    recognize