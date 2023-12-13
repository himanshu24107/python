def main():
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
                    

            
