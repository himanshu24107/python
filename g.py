from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
from function import *
from constants import *

app = Flask(__name__)

# Video capture object
cap = cv2.VideoCapture(0)

# Set up MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Global variable to store the current frame
global_frame = None

# Set the frame skip interval
frame_skip_interval = 2  # Process every second frame

def generate_frames():
    global global_frame
    while True:
        success, frame = cap.read()

        if not success:
            break
        else:
            # Update the global frame variable
            global_frame = frame

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def process_frames():
    global global_frame
    frame_count = 0

    while True:
        if global_frame is not None:
            frame_count += 1

            if frame_count % frame_skip_interval == 0:
                frame = global_frame

                # Convert the image to RGB for processing by MediaPipe Pose
                with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Process the frame with MediaPipe Pose
                    results = pose.process(frame_rgb)

                    # Clear the frame with the black background
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
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
                    frame = frame_bgr

                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_combined')
def video_combined():
    return Response(process_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
