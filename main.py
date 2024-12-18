import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import os


PIXEL_TO_METER = 0.001  #for speed

# Our tained YOLOv8 model
model = YOLO("tabletennisbest.pt")

output_dir = os.path.dirname(os.path.abspath(__file__))

video_path = "c05.mov"  # Video we want to test

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS: {fps}") 

# Initialize variables for trajectory and speedq
trajectory_2d = []  
speeds = []         

#Save the processed video
video_writer = cv2.VideoWriter(os.path.join(output_dir, "speed_estimation_output.avi"),
                               cv2.VideoWriter_fourcc(*"mp4v"), fps, 
                               (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# Creating a 3D plot for trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


scale_factor = 0.3  # Size of the display

frame_count = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    # YOLO detection
    results = model.predict(frame)
    ball_position = None

    for result in results:
        for bbox in result.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = bbox
            if cls == 0 and conf > 0.5:  # Class 0: Ball
                # Calculate ball center and bounding box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                ball_position = (center_x, center_y)
                trajectory_2d.append(ball_position)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
                
                # Calculate speed
                if len(trajectory_2d) > 1:
                    prev_x, prev_y = trajectory_2d[-2]
                    curr_x, curr_y = trajectory_2d[-1]
                    
                    # Calculate instantaneous speed (meters/second)
                    distance = ((curr_x - prev_x)**2 + (curr_y - prev_y)**2)**0.5 * PIXEL_TO_METER
                    instantaneous_speed = distance * fps  

                    # Convert speed to miles per hour
                    speed_mph = instantaneous_speed * 2.23694 
                    speeds.append(speed_mph)

                    # Show speed 
                    speed_text = f"{speed_mph:.2f} mph"
                    cv2.putText(frame, speed_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

                cv2.putText(frame, "Ball", (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                break

    resized_frame = cv2.resize(frame, (int(frame.shape[1] * scale_factor), int(frame.shape[0] * scale_factor)))

    cv2.imshow("Ball Tracking and Speed", resized_frame)
    video_writer.write(frame) 
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
video_writer.release()  
cv2.destroyAllWindows()

# Print the speed values to check them
print(f"Speeds (mph): {speeds}")

# 3D trajectory of the ball
trajectory_2d = np.array(trajectory_2d)
if len(trajectory_2d) > 0:
    ax.plot(trajectory_2d[:, 0], trajectory_2d[:, 1], zs=frame_count, label='Ball Trajectory')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Frames')
    ax.set_title('3D Trajectory of the Ball')

    # Saving the 3D plot as an image file after running
    plot_file_path = os.path.join(output_dir, "ball_trajectory_3d.png")
    plt.savefig(plot_file_path)
    print(f"3D trajectory plot saved as {plot_file_path}")

    plt.show()
