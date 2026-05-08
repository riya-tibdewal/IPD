# from ultralytics import YOLO
# import cv2
# import mediapipe as mp
# import os

# # Load YOLO model
# model = YOLO("yolov8n.pt")

# # MediaPipe Pose
# mp_pose = mp.solutions.pose
# mp_draw = mp.solutions.drawing_utils

# pose = mp_pose.Pose()

# # Input video folder
# video_folder = "../videos"

# # Output folder
# output_folder = "../output"

# # Create output folder if not exists
# os.makedirs(output_folder, exist_ok=True)

# # Loop through all videos
# for video_name in os.listdir(video_folder):

#     video_path = os.path.join(video_folder, video_name)

#     print(f"Processing: {video_name}")

#     # Open video
#     cap = cv2.VideoCapture(video_path)

#     # Get video properties
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))

#     # Output video path
#     output_path = os.path.join(
#         output_folder,
#         f"output_{video_name}"
#     )

#     # Video writer
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')

#     out = cv2.VideoWriter(
#         output_path,
#         fourcc,
#         fps,
#         (width, height)
#     )
#     frame_count = 0

#     while True:

#         ret, frame = cap.read()

#         if not ret:
#             break
#         frame_count += 1
#         if frame_count % 5 != 0:
#           continue
#         # YOLO Detection
#         results = model(frame)

#         annotated_frame = results[0].plot()

#         # Convert frame for MediaPipe
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Pose Detection
#         pose_results = pose.process(rgb)

#         # Draw skeleton
#         if pose_results.pose_landmarks:

#             mp_draw.draw_landmarks(
#                 annotated_frame,
#                 pose_results.pose_landmarks,
#                 mp_pose.POSE_CONNECTIONS
#             )

#         # Show live output
#         cv2.imshow("Detection", annotated_frame)

#         # Save frame
#         out.write(annotated_frame)

#         # Press q to stop
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release resources
#     cap.release()
#     out.release()

#     print(f"Saved: {output_path}")

# # Close all windows
# cv2.destroyAllWindows()

# print("ALL VIDEOS PROCESSED SUCCESSFULLY")



from ultralytics import YOLO
import cv2
import mediapipe as mp
import os

# Load YOLO model
model = YOLO("yolov8n.pt")

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose()

# Input folder
video_folder = "../videos"

# Output folder
output_folder = "../output"

# Create output folder
os.makedirs(output_folder, exist_ok=True)

# Loop through videos
for video_name in os.listdir(video_folder):

    video_path = os.path.join(video_folder, video_name)

    print(f"Processing: {video_name}")

    # Open video
    cap = cv2.VideoCapture(video_path)

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Output video path
    output_path = os.path.join(
        output_folder,
        f"output_{video_name}"
    )

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Process every 5th frame
        if frame_count % 5 != 0:
            continue

        # YOLO detection
        results = model(frame)

        # KEEP ORIGINAL YOLO OUTPUT
        annotated_frame = results[0].plot()

        # Loop through detections
        for result in results:

            boxes = result.boxes

            for box in boxes:

                # Detect only PERSON class
                cls = int(box.cls[0])

                if cls == 0:

                    confidence = float(box.conf[0])

                    if confidence < 0.5:
                        continue

                    # Bounding box
                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    # Crop person
                    person_crop = frame[y1:y2, x1:x2]

                    if person_crop.size == 0:
                        continue

                    try:

                        # Convert to RGB
                        rgb_crop = cv2.cvtColor(
                            person_crop,
                            cv2.COLOR_BGR2RGB
                        )

                        # Pose detection
                        pose_results = pose.process(rgb_crop)

                        # Draw skeleton manually
                        if pose_results.pose_landmarks:

                            h, w, _ = person_crop.shape

                            # Draw connections
                            for connection in mp_pose.POSE_CONNECTIONS:

                                start_idx = connection[0]
                                end_idx = connection[1]

                                start = pose_results.pose_landmarks.landmark[start_idx]
                                end = pose_results.pose_landmarks.landmark[end_idx]

                                x_start = int(start.x * w) + x1
                                y_start = int(start.y * h) + y1

                                x_end = int(end.x * w) + x1
                                y_end = int(end.y * h) + y1

                                cv2.line(
                                    annotated_frame,
                                    (x_start, y_start),
                                    (x_end, y_end),
                                    (0, 255, 0),
                                    2
                                )

                            # Draw keypoints
                            for landmark in pose_results.pose_landmarks.landmark:

                                cx = int(landmark.x * w) + x1
                                cy = int(landmark.y * h) + y1

                                cv2.circle(
                                    annotated_frame,
                                    (cx, cy),
                                    3,
                                    (0, 0, 255),
                                    -1
                                )

                    except:
                        pass

        # Show output
        cv2.imshow(
            "YOLO + Multi Person Pose",
            annotated_frame
        )

        # Save output
        out.write(annotated_frame)

        # Quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release
    cap.release()
    out.release()

    print(f"Saved: {output_path}")

# Close windows
cv2.destroyAllWindows()

print("ALL VIDEOS PROCESSED SUCCESSFULLY")




