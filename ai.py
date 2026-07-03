import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ========================
# Device setup
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========================
# Load YOLOv8 model
# ========================
model = YOLO("yolov8n.pt")  # yolov8s.pt is faster
model.to(device)

# ========================
# DeepSORT tracker
# ========================
tracker = DeepSort(max_age=20, max_cosine_distance=0.2, nn_budget=50)

# ========================
# Vehicle classes (COCO IDs)
# ========================
vehicle_classes = [1, 2, 3, 5, 7]  # Bicycle, Car, Motorcycle, Bus, Truck
vehicle_class_names = {
    1: "Bicycle", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"
}

# ========================
# Video input
# ========================
video_path = "s.mp4"
cap = cv2.VideoCapture(video_path)


# ========================
# Line drawing (legs)
# ========================
lines = []
drawing = False
current_line = []
typing_name = False
current_name = ""
mid_point = (0, 0)

# ========================
# Counting data
# ========================
track_histories = {}  # track_id -> list of (x,y)
vehicle_entry = {}   # track_id -> entry_leg
movement_counts = {} # (entry, exit) -> count
legs = []            # store all leg names (for matrix display)

# Frame skipping
frame_counter = 0
skip_rate = 1  # process every 2nd frame (adjust for speed)

# ========================
# Helper functions
# ========================
def get_line_direction(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    angle = np.degrees(np.arctan2(dy, dx)) % 360
    if 45 <= angle < 135:
        return "Top_to_Bottom"
    if 225 <= angle < 315:
        return "Bottom_to_Top"
    if 135 <= angle < 225:
        return "Right_to_Left"
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
            mid_point = (
                (current_line[0][0] + current_line[1][0]) // 2,
                (current_line[0][1] + current_line[1][1]) // 2,
            )
            typing_name = True
            current_name = ""

def draw_lines(frame):
    for pt1, pt2, _, name in lines:
        cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
        mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        cv2.putText(frame, name, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

# Segment intersection
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def check_segment_intersection(p1, p2, q1, q2):
    return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))

cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Video", mouse_callback)

# ========================
# Precompute line bounding boxes
# ========================
def compute_line_bboxes():
    return [(min(p1[0], p2[0]), min(p1[1], p2[1]),
             max(p1[0], p2[0]), max(p1[1], p2[1])) for p1, p2, _, _ in lines]

line_bboxes = compute_line_bboxes()


# def classify_turn(trajectory):
#     if len(trajectory) < 10:
#         return None

#     p1 = trajectory[0]
#     p2 = trajectory[len(trajectory)//2]
#     p3 = trajectory[-1]

#     v1 = (p2[0]-p1[0], p2[1]-p1[1])
#     v2 = (p3[0]-p2[0], p3[1]-p2[1])

#     angle1 = np.arctan2(v1[1], v1[0])
#     angle2 = np.arctan2(v2[1], v2[0])

#     angle_diff = np.degrees(angle2 - angle1)

#     if angle_diff > 20:
#         return "Left Turn"
#     elif angle_diff < -20:
#         return "Right Turn"
#     else:
#         return "Straight"

# ========================
# Main loop
# ========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    if frame_counter % skip_rate != 0:
        continue

    # ========================
    # YOLO detection (optimized)
    # ========================
    results = model(frame, imgsz=1280, conf=0.5, iou=0.6, device=device, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    cls_ids = results.boxes.cls.cpu().numpy().astype(int)
    confs = results.boxes.conf.cpu().numpy()

    detections = []
    for (x1, y1, x2, y2), cls_id, conf in zip(boxes, cls_ids, confs):
        if cls_id in vehicle_classes:
            w, h = int(x2-x1), int(y2-y1)
            if w*h >= 500:
                detections.append(([int(x1), int(y1), w, h], float(conf), cls_id))

    # Pass class_id as metadata to DeepSORT
    tracks = tracker.update_tracks(detections, frame=frame)

    # ========================
    # Update tracks
    # ========================
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        class_id = getattr(track, "det_class", None)

        trajectory = track_histories.get(track_id, [])

        curr_pos = (cx, cy)

        if len(trajectory) > 0:
            prev_pos = trajectory[-1]   # last stored point
        else:
            prev_pos = curr_pos

        # ========================
        # Check crossing with lines (optimized)
        # ========================
        line_bboxes = compute_line_bboxes()
        for (lx1, ly1, lx2, ly2), (lp1, lp2, _, name) in zip(line_bboxes, lines):
            if not (lx1-5 <= cx <= lx2+5 and ly1-5 <= cy <= ly2+5):
                continue
            if check_segment_intersection(prev_pos, curr_pos, lp1, lp2):
                if track_id not in vehicle_entry:
                    vehicle_entry[track_id] = name
                else:
                    entry = vehicle_entry[track_id]
                    exit_leg = name
                    if entry != exit_leg:
                        trajectory = track_histories.get(track_id, [])
                        # movement_type = classify_turn(trajectory)

                        # if movement_type:
                        movement_counts[(entry, exit_leg)] = \
                            movement_counts.get((entry, exit_leg), 0) + 1    
                        if entry not in legs:
                            legs.append(entry)
                        if exit_leg not in legs:
                            legs.append(exit_leg)
                        del vehicle_entry[track_id]

        if track_id not in track_histories:
            track_histories[track_id] = []

        track_histories[track_id].append(curr_pos)

        # Keep only last 20 points
        if len(track_histories[track_id]) > 20:
            track_histories[track_id].pop(0)

        # Draw box + ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
        if class_id is not None:
            label = f"{vehicle_class_names[class_id]} #{track_id}"
            cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Draw static lines
    draw_lines(frame)

    # Draw current line while creating
    if drawing and len(current_line) == 1:
        cv2.circle(frame, current_line[0], 5, (0, 0, 255), -1)
    elif typing_name and len(current_line) == 2:
        cv2.line(frame, current_line[0], current_line[1], (0, 255, 255), 2)
        cv2.putText(frame, current_name, (mid_point[0]-40, mid_point[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # ========================
    # Dynamic TMC Matrix (every 3 frames)
    # ========================
    if legs and frame_counter % 3 == 0:
        sorted_legs = sorted(legs)
        start_x, start_y = 30, 50
        cell_w, cell_h = 80, 30

        # Header
        cv2.putText(frame, "TMC Matrix", (start_x, start_y-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        for j, exit_leg in enumerate(sorted_legs):
            cv2.putText(frame, str(exit_leg), (start_x + (j+1)*cell_w, start_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Rows
        for i, entry_leg in enumerate(sorted_legs):
            y = start_y + (i+1)*cell_h
            cv2.putText(frame, str(entry_leg), (start_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            for j, exit_leg in enumerate(sorted_legs):
                x = start_x + (j+1)*cell_w
                count = movement_counts.get((entry_leg, exit_leg), 0)
                cv2.putText(frame, str(count), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Video", frame)
    key = cv2.waitKey(1) & 0xFF

    # ========================
    # Line naming input
    # ========================
    if typing_name:
        if key == 13:  # Enter
            direction = get_line_direction(current_line[0], current_line[1])
            lines.append((current_line[0], current_line[1], direction, current_name))
            if current_name not in legs:
                legs.append(current_name)
            drawing = False
            typing_name = False
            current_name = ""
            current_line = []
        elif key == 8:  # Backspace
            current_name = current_name[:-1]
        elif 32 <= key <= 126:
            current_name += chr(key)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# ========================
# Final report
# ========================
print("\n=== Turning Movement Counts ===")
for (entry, exit_leg), count in movement_counts.items():
    print(f"{entry} -> {exit_leg} : {count}")
print("================================\n")
