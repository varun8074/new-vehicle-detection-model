from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

# Load YOLOv8 model (smallest/lightest by default)
model = YOLO('yolov8n.pt')

# Vehicle classes from COCO dataset
vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# Load video
video_path = 'a.mp4'
cap = cv2.VideoCapture(video_path)

cv2.namedWindow('DeepSORT Vehicle Tracking', cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model(frame)[0]

    detections = []

    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        class_id = int(cls)

        if class_id in vehicle_classes:
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf.item(), class_id))

    # Update DeepSORT tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw results
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow('DeepSORT Vehicle Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
