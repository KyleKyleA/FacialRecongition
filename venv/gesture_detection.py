# Starter Code for Gesture Detection using MediaPipe and OpenCV from Google MediaPipe's documentation: https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer/python
# Youtube Video's tutorials and etc.
# looking forward to learning and implementing something unique as I dive into a bit of machine learning and AI in this GITHUB repository.
# IMPORTS
import cv2
import mediapipe as mp
import time
import psutil
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# for static images
IMAGE_FILES = []

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:

    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.flip(cv2.imread(file), 1)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        print('handedness:', results.multi_handedness)
        if not results.multi_hand_landmarks:
            continue
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            print('hand_landmarks:', hand_landmarks)
            print(
                f'index finger tip coordinates: ('
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
            )
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
        if not results.multi_hand_world_landmarks:
            continue
        for hand_world_landmarks in results.multi_hand_world_landmarks:
            mp_drawing.plot_landmarks(
                hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

# WEB CAM
cap = cv2.VideoCapture(0)
prev_time = 0

# Setting width and height for mediapipe 
WIDTH = 640
HEIGHT = 480


with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        resized_image = cv2.resize(image, (WIDTH, HEIGHT))
        
        
        image.flags.writeable = False
        image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # Displaying fps and current time on the video feed 
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        
        # Text on the application 
        cv2.putText(image, time.strftime('%Y-%m-%d'), (10, 30), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
        cv2.putText(image, f'FPS: {int(fps)}', (10, 70), cv2.FONT_ITALIC, 1, (255, 0, 0), 3)
        
        # Displaying CPU and Memory Usage on the video feed
        cpu_pct = psutil.cpu_percent()

        # RAM
        ram = psutil.virtual_memory()
        ram_pct = ram.percent
        ram_used_in_gb = round(ram.used / (1024 ** 3), 2)

        # Text on the application for ram and cpu usage 
        cv2.putText(image, f'CPU: {cpu_pct}%', (10, 110), cv2.FONT_ITALIC, 1, (255, 0, 0), 3)
        cv2.putText(image, f'RAM: {ram_pct}% ({ram_used_in_gb} GB)', (10, 150), cv2.FONT_ITALIC, 1, (255, 0, 0), 3)

        
        # Show Hands 
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
        


# Running application
cap.release()
cv2.destroyAllWindows()