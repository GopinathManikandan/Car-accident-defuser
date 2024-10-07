import numpy as np
import dlib
import cv2
import time
import pygame

# Initialize pygame mixer for playing alert sound
pygame.mixer.init()
pygame.mixer.music.load('emergency_alert (1).mp3')

# Constants
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
EYES = list(range(36, 48))
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TITLE_NAME = 'Face Drowsiness Detection'

FACE_CASCADE_PATH = './haarcascades/haarcascade_frontalface_alt.xml'
PREDICTOR_PATH = './model/shape_predictor_68_face_landmarks.dat'

MIN_EAR_THRESHOLD = 0.25
CLOSED_EYE_FRAME_LIMIT = 7
ALERT_COLOR_AWAKE = (0, 255, 0)
ALERT_COLOR_SLEEP = (0, 0, 255)

# Load face detection and landmark predictor models
face_cascade = cv2.CascadeClassifier()
if not face_cascade.load(cv2.samples.findFile(FACE_CASCADE_PATH)):
    print('--(!)Error loading face cascade')
    exit(0)

predictor = dlib.shape_predictor(PREDICTOR_PATH)

status = 'Awake'
number_closed = 0
show_frame = None
sign = None
color = ALERT_COLOR_AWAKE


def getEAR(points):
    A = np.linalg.norm(points[1] - points[5])
    B = np.linalg.norm(points[2] - points[4])
    C = np.linalg.norm(points[0] - points[3])
    return (A + B) / (2.0 * C)


def detectAndDisplay(image):
    global number_closed, color, show_frame, sign, status

    start_time = time.time()
    image = cv2.resize(image, (FRAME_WIDTH, FRAME_HEIGHT))
    show_frame = image
    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    faces = face_cascade.detectMultiScale(frame_gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), ALERT_COLOR_AWAKE, 2)

        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        points = np.matrix([[p.x, p.y] for p in predictor(frame_gray, rect).parts()])
        show_parts = points[EYES]
        right_eye_EAR = getEAR(points[RIGHT_EYE])
        left_eye_EAR = getEAR(points[LEFT_EYE])
        mean_eye_EAR = (right_eye_EAR + left_eye_EAR) / 2

        right_eye_center = np.mean(points[RIGHT_EYE], axis=0).astype("int")
        left_eye_center = np.mean(points[LEFT_EYE], axis=0).astype("int")

        cv2.putText(image, "{:.2f}".format(right_eye_EAR), (right_eye_center[0, 0], right_eye_center[0, 1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, ALERT_COLOR_AWAKE, 1)
        cv2.putText(image, "{:.2f}".format(left_eye_EAR), (left_eye_center[0, 0], left_eye_center[0, 1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, ALERT_COLOR_AWAKE, 1)

        for (i, point) in enumerate(show_parts):
            x = point[0, 0]
            y = point[0, 1]
            cv2.circle(image, (x, y), 1, (0, 255, 255), -1)

        if mean_eye_EAR > MIN_EAR_THRESHOLD:
            color = ALERT_COLOR_AWAKE
            status = 'Awake'
            number_closed = max(number_closed - 1, 0)
            # Stop playing the sound if the person is awake
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
        else:
            color = ALERT_COLOR_SLEEP
            status = 'Sleep'
            number_closed += 1
            # Play alert sound if drowsiness detected and audio is not already playing
            if number_closed > CLOSED_EYE_FRAME_LIMIT and not pygame.mixer.music.get_busy():
                pygame.mixer.music.play()

        sign = f"{status}, Sleep count: {number_closed} / {CLOSED_EYE_FRAME_LIMIT}"

    cv2.putText(show_frame, sign, (10, FRAME_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imshow(TITLE_NAME, show_frame)

    frame_time = time.time() - start_time
    print(f"Frame time: {frame_time:.3f} seconds")


# Start video capture
vs = cv2.VideoCapture(0)
time.sleep(2.0)
if not vs.isOpened():
    print('### Error opening video ###')
    exit(0)

while True:
    ret, frame = vs.read()
    if frame is None:
        print('### No more frame ###')
        vs.release()
        break
    detectAndDisplay(frame)
    if cv2.waitKey(1) & 0xFF == ord('r'):
        break

vs.release()
cv2.destroyAllWindows()
