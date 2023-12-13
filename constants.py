from function import *
from constants import *
import mediapipe as mp
import random
import time
from main import *
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Define stick figure keypoints
stick_figure = {
    "left_arm": [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST],
    "right_arm": [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST],
    "left_leg": [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE],
    "right_leg": [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE],
    "shoulders": [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER],
    "waist": [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP],
    "left": [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP],
    "right": [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP],
}


# Define black background
background_color = (0, 0, 0)  # Black

# Define white color for stick figure
drawing_color = (255, 255,0)  # colour

# Visibility threshold for landmarks
visibility_threshold = 0.4

# Blinking state flag
is_eye_blinking = False

# Blinking duration and interval
blink_duration = random.uniform(0.09, 0.2)
blink_interval = random.uniform(3, 10)

# Blinking timer
blink_timer = time.time() + blink_interval

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))


frame_delay = int(1000 / fps)


angle_min = []
angle_min_hip = []

# Curl counter variables
counter = 0 

# Global variable to control video processing
process_frame = False
