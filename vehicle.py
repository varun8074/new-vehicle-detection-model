import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# DeepSORT tracker with tuned parameters
tracker = DeepSort(max_age=30, max_cosine_distance=0.2, nn_budget=100)

# Classes to track (vehicles + person, bicycle)
vehicle_classes = [0, 1, 2, 3, 5, 7]
vehicle_class_names = {
    0: 'Person',
    1: 'Bicycle',
    2: 'Car',
    3: 'Motorcycle',
    5: 'Bus',
    7: 'Truck'
}

# Video source
video_path = 'model/s.mp4'
cap = cv2.VideoCapture(video_path)

# Lines and counting variables
lines = []
drawing = False
current_line = []
typing_name = False
current_name = ""
mid_point = (0, 0)

track_histories = {}
counts_per_line = []
counted_ids_per_line = []

# Frame skip settings
frame_counter = 0
skip_rate = 1  # process every 2nd frame

def get_line_direction(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    angle = np.degrees(np.arctan2(dy, dx)) % 360
    if 45 <= angle < 135:
        return "Top_to_Bottom"
    elif 225 <= angle < 315:
        return "Bottom_to_Top"
    elif 135 <= angle < 225:
        return "Right_to_Left"
    else:
        return "Left_to_Right"

def mouse_callback(event, x, y, flags, param):
    global drawing, current_line, lines, typing_name, mid_point, current_name
    if typing_name:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        if not drawing:
            drawing = True
            current_line = [(x, y)]
        elif drawing and len(current_line) == 1:
            current_line.append((x, y))
            mid_point = ((current_line[0][0] + current_line[1][0]) // 2, (current_line[0][1] + current_line[1][1]) // 2)
            typing_name = True
            current_name = ""

def draw_lines(frame):
    for pt1, pt2, _, _ in lines:
        cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

def check_line_crossing(p1, p2, line_p1, line_p2):
    def side(a, b, c):
        return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])
    side1 = side(line_p1, line_p2, p1)
    side2 = side(line_p1, line_p2, p2)
    return side1 * side2 < 0

cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Video', mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    if frame_counter % skip_rate != 0:
        continue

    # Run YOLO detection
    results = model(frame, device=device, verbose=False)[0]

    boxes = []
    confidences = []
    class_ids = []

    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        if conf.item() < 0.6:  # Increased confidence threshold
            continue
        class_id = int(cls)
        if class_id in vehicle_classes:
            x1, y1, x2, y2 = map(int, box)
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confidences.append(float(conf))
            class_ids.append(class_id)

    # Apply NMS to reduce overlapping detections
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.6, nms_threshold=0.4)
    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            detections.append((boxes[i], confidences[i], class_ids[i]))

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # Find class_id for current track from detections close to center
        class_id = None
        for det in detections:
            dx, dy, dw, dh = det[0]
            dcx, dcy = dx + dw / 2, dy + dh / 2
            if abs(dcx - cx) < 20 and abs(dcy - cy) < 20:
                class_id = det[2]
                break

        prev_pos = track_histories.get(track_id, (cx, cy))
        curr_pos = (cx, cy)

        # Count crossings
        for i, (lp1, lp2, _, _) in enumerate(lines):
            if track_id not in counted_ids_per_line[i]:
                if check_line_crossing(prev_pos, curr_pos, lp1, lp2):
                    counts_per_line[i] += 1
                    counted_ids_per_line[i].add(track_id)

        track_histories[track_id] = curr_pos

        # Draw bounding box and center
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

        # Draw label if class is known
        if class_id is not None:
            label = f"{vehicle_class_names[class_id]} #{track_id}"
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Draw lines and labels
    draw_lines(frame)

    if drawing and len(current_line) == 1:
        cv2.circle(frame, current_line[0], 5, (0, 0, 255), -1)
    elif typing_name and len(current_line) == 2:
        cv2.line(frame, current_line[0], current_line[1], (0, 255, 255), 2)
        cv2.putText(frame, current_name, (mid_point[0] - 40, mid_point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    for i, (lp1, lp2, direction, name) in enumerate(lines):
        mid = ((lp1[0] + lp2[0]) // 2, (lp1[1] + lp2[1]) // 2)
        label = f"{name} ({direction}): {counts_per_line[i]}"
        cv2.putText(frame, label, (mid[0] - 50, mid[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow('Video', frame)
    key = cv2.waitKey(1) & 0xFF

    if typing_name:
        if key == 13:  # Enter key confirms line name
            direction = get_line_direction(current_line[0], current_line[1])
            lines.append((current_line[0], current_line[1], direction, current_name))
            counts_per_line.append(0)
            counted_ids_per_line.append(set())
            drawing = False
            typing_name = False
            current_name = ""
            current_line = []
        elif key == 8:  # Backspace deletes last char
            current_name = current_name[:-1]
        elif 32 <= key <= 126:  # Printable chars
            current_name += chr(key)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Final report
print("\n=== Vehicle Count Summary ===")
for i, (_, _, direction, name) in enumerate(lines):
    print(f"Line '{name}' ({direction}) : {counts_per_line[i]} crossings")
print("================================\n")
