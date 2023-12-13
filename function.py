from constants import *

def draw_stick_figure(frame_bgr, joint_points):
    if len(joint_points) >= 2:
        # Draw lines connecting the keypoints in white color
        for i in range(len(joint_points) - 1):
            try:
                cv2.line(frame_bgr, joint_points[i], joint_points[i+1], drawing_color, 10)
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
