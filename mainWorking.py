import cv2
import mediapipe as mp
import time
import random
import numpy as np
# Set up webcam video capture
cap = cv2.VideoCapture(0)

# Set up MediaPipe Pose
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

def draw_stick_figure(frame_bgr, joint_points):
    if len(joint_points) >= 2:
        # Draw lines connecting the keypoints in white color
        for i in range(len(joint_points) - 1):
            try:
                cv2.line(frame_bgr, joint_points[i], joint_points[i+1], drawing_color, 5)
            except:
                pass

def draw_head(frame_bgr, results, joint_points):
    if results.pose_landmarks and results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].visibility > visibility_threshold:
        nose_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * frame_bgr.shape[1])
        nose_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * frame_bgr.shape[0])

        if joint_points and len(joint_points) > 0:
            radius = int((joint_points[0][1] - nose_y) * 0.7)
            if radius >= 0:
                cv2.circle(frame_bgr, (nose_x, nose_y), radius, drawing_color, 2)

        # Draw eyes or hide them during blinking
        if not is_eye_blinking:
            left_eye_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE]
            right_eye_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE]

            if left_eye_landmark.visibility > visibility_threshold:
                left_eye_x = int(left_eye_landmark.x * frame_bgr.shape[1])
                left_eye_y = int(left_eye_landmark.y * frame_bgr.shape[0])
                eye_radius = int((joint_points[0][1] - nose_y) * 0.1)
                cv2.line(frame_bgr, (left_eye_x, left_eye_y - eye_radius), (left_eye_x, left_eye_y + eye_radius), drawing_color, 2)

            if right_eye_landmark.visibility > visibility_threshold:
                right_eye_x = int(right_eye_landmark.x * frame_bgr.shape[1])
                right_eye_y = int(right_eye_landmark.y * frame_bgr.shape[0])
                eye_radius = int((joint_points[0][1] - nose_y) * 0.1)
                cv2.line(frame_bgr, (right_eye_x, right_eye_y - eye_radius), (right_eye_x, right_eye_y + eye_radius), drawing_color, 2)

         # Draw circular smile
        smile_radius = int((joint_points[0][1] - nose_y) * 0.3)
        smile_center_x = nose_x
        smile_center_y = nose_y + int((joint_points[0][1] - nose_y) * 0.1)

        start_angle = 30
        end_angle = 155

        # Draw circular arc for smile
        try:
            cv2.ellipse(frame_bgr, (smile_center_x, smile_center_y), (smile_radius, smile_radius), 0, start_angle, end_angle, drawing_color, 2)
        except:
            pass


def update_blink_state():
    global is_eye_blinking, blink_timer

    if time.time() >= blink_timer:
        is_eye_blinking = not is_eye_blinking
        if is_eye_blinking:
            blink_timer = time.time() + blink_duration
        else:
            blink_timer = time.time() + blink_interval



# grab the width, height, and fps of the frames in the video stream.
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(fps)

# initialize the FourCC and a video writer object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))


process_frame = True

frame_delay = int(1000 / fps)

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

def rescale_frame(frame, percent=50):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

angle_min = []
angle_min_hip = []

# Curl counter variables
counter = 0 
def main():
    try:
        # Main program loop
        with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
            while cap.isOpened():
                # Read webcam frame
                ret, frame = cap.read()
                
                if not ret:
                    print("Can't receive frame (stream end?). Exiting...")
                    break
                if process_frame:
                    # Convert the image to RGB for processing by MediaPipe Pose
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Process the frame with MediaPipe Pose
                    results = pose.process(frame_rgb)

                    # Clear the frame with the black background
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    frame_bgr[:] = background_color

                    # Initialize joint points as an empty list
                    joint_points = []

                    # Draw the stick figure on the frame
                    if results.pose_landmarks:
                        for part in stick_figure.values():
                            joint_points = []
                            for landmark in part:
                                if results.pose_landmarks.landmark[landmark].visibility > visibility_threshold:
                                    joint_points.append(mp_drawing._normalized_to_pixel_coordinates(
                                        results.pose_landmarks.landmark[landmark].x,
                                        results.pose_landmarks.landmark[landmark].y,
                                        frame_bgr.shape[1], frame_bgr.shape[0]))

                            draw_stick_figure(frame_bgr, joint_points)

                    draw_head(frame_bgr, results, joint_points)

                    # Update blinking state
                    update_blink_state()
                    
                    # Flip the frame horizontally
                    frame_bgr = cv2.flip(frame_bgr, 1)
                    # Set the desired window width and height
                    window_width = 800
                    window_height = 600
                    

                    # Calculate the aspect ratio of the frame
                    frame_aspect_ratio = frame_width / frame_height

                    # Calculate the corresponding window width and height to maintain the aspect ratio
                    if window_width / window_height > frame_aspect_ratio:
                        window_width = int(window_height * frame_aspect_ratio)
                    else:
                        window_height = int(window_width / frame_aspect_ratio)

                    # Create the named window with the calculated width and height
                    cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Webcam', window_width, window_height)


                    try:
                        landmarks = results.pose_landmarks.landmark
                    
                        # Get coordinates
                        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    
                    
                        # Get coordinates
                        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                        knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                        ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    
                        # Calculate angle
                        angle = calculate_angle(shoulder, elbow, wrist)
                        print(f'Elbow: {round(angle, 0)}')

                        angle_knee = calculate_angle(hip, knee, ankle) 
                        print(f'Knee : {round(angle_knee,0)}')
                        
                        angle_hip = calculate_angle(shoulder, hip, knee)
                        angle_hip = round(angle_hip,0)
                        
                        # hip_angle = 180-angle_hip
                        # knee_angle = 180-angle_knee
                        
                        
                        # angle_min.append(angle_knee)
                        # angle_min_hip.append(angle_hip)
                        
                        # Visualize angle
                        cv2.putText(frame_bgr, str(angle), 
                                    tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA
                                            )
                                
                            
                        cv2.putText(frame_bgr, str(angle_knee), 
                                    tuple(np.multiply(knee, [1500, 800]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA
                                            )
                        
                        cv2.putText(frame_bgr, str(angle_hip), 
                                    tuple(np.multiply(hip, [1500, 800]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA
                                            )
                    except:
                            pass
                    # Inside the while loop:
                    cv2.imshow('Webcam', frame_bgr)
                        
                    output.write(frame_bgr)
                    
        # process_frame = not process_frame
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except:
        print('recalibrating....')
        main()
main()

# Release the webcam and destroy all windows
cap.release()
output.release()
cv2.destroyAllWindows()
