from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp
from function import *
from constants import *
import nltk
import os
from go import *
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
frame_count = 0

# Initialize nltk (you may want to download additional resources if needed)
nltk.download('punkt')

# Example usage:
target_directory = "C:/Users/DELL/Desktop/python/data"
query_result = "No result yet"
lst=[]
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
    
    while True:
        # if file is not None:
        #     cap = cv2.VideoCapture(file)
        #     success, frame = cap.read()
        #     if not success:
        #         break
        #     else:
        #         # Update the global frame variable
        #         global_frame = frame
        # else:
        #     pass

        if global_frame is not None:
            frame = global_frame

            # Convert the image to RGB for processing by MediaPipe Pose
            with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process the frame with MediaPipe Pose
                results = pose.process(frame)

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
                frame = cv2.flip(frame_bgr, 1)

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def proc(file):
    global global_frame
    cap = cv2.VideoCapture(f'C:/Users/DELL/Desktop/python/data/{file}')
    while True:
        # cap = cv2.VideoCapture(f'C:/Users/DELL/Desktop/python/data/{file}')
        success, frame = cap.read()
        print(frame)
        if not success:
            break
        else:
         # Update the global frame variable
            global_frame = frame
            print('inside')
        

            if global_frame is not None:
                frame = global_frame

                # Convert the image to RGB for processing by MediaPipe Pose
                with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Process the frame with MediaPipe Pose
                    results = pose.process(frame)

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
                    frame = cv2.flip(frame_bgr, 1)

                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed', methods = ['GET'])
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_combined', methods = ['GET'])
def video_combined():
    return Response(process_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/select_option', methods=['POST'])
def select_option():
    selected_option = request.form.get('option')

    if selected_option == 'text':
        return render_template('text_input.html', option='text')

    if selected_option == 'video':
        return render_template('video_combined.html', option ='video')

    return render_template('index.html', option='video')

@app.route('/process_text', methods=['POST'])
def process_text():
    global query_result
    # Get text input from the form
    user_input = request.form.get('user_input')

    # Perform text processing here (replace with your actual text processing logic)
    matching_files = search_similar_files(target_directory, user_input)

    if matching_files:
        matching_files.sort(key=lambda x: x[1], reverse=True)
        result_string = "Similar Files:\n"
        for file, similarity in matching_files:
            result_string += f"{file} (Similarity: {similarity:.2f})\n"
            lst.append(file)
            break
            
    else:
        result_string = "No similar files found."
        

    # Update the global variable with the result
    query_result = result_string

    return render_template('text_result.html', option='text', result=query_result)

@app.route('/movie', methods=['GET'])
def movie():
    return Response(proc(lst[0]), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
