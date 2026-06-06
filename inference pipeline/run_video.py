# Usage -> python E:\Drone\IRDDT\pipeline\run_video.py --split train --seq 20190925_152412_1_4

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from ultralytics import YOLO
from tracker import Tracker, ACTIVE, LOST, ABSENT, RE_ACQUIRED


# Different color for each track ID
TRACK_COLORS = [
    (0, 255, 0),      # Green
    (255, 165, 0),    # Orange
    (0, 200, 255),    # Gold
    (255, 0, 128),    # Pink
    (0, 255, 255),    # Yellow
    (255, 200, 100),  # Light blue
    (100, 255, 100),  # Light green
    (200, 100, 255),  # Purple
    (100, 200, 200),  # Tan
    (255, 100, 100),  # Blue
]

STATE_LABELS = {
    ACTIVE:      "TRACKING",
    LOST:        "LOST",
    ABSENT:      "ABSENT",
    RE_ACQUIRED: "RE-ACQUIRED",
}

# Dimmed versions for LOST/ABSENT states
def dim_color(color, factor=0.4):
    return tuple(int(c * factor) for c in color)


def get_track_color(track_id, state):
    # get color for a track — unique per ID, dimmed when LOST/ABSENT.
    base = TRACK_COLORS[track_id % len(TRACK_COLORS)]
    if state in (LOST, ABSENT):
        return dim_color(base)
    return base


def draw_tracks(frame, tracks, frame_num, fps_val):
    # draw tracking visualization on frame.
    h, w = frame.shape[:2]

    for t in tracks:
        bbox = t["bbox"]  # [cx, cy, w, h] in pixel coords
        state = t["state"]
        color = get_track_color(t["track_id"], state)

        # Convert center format to corners
        x1 = int(bbox[0] - bbox[2] / 2)
        y1 = int(bbox[1] - bbox[3] / 2)
        x2 = int(bbox[0] + bbox[2] / 2)
        y2 = int(bbox[1] + bbox[3] / 2)

        # Draw bounding box
        thickness = 4 if state == RE_ACQUIRED else (2 if state == ACTIVE else 1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Draw label: "TRACKING 0.87"
        label = f"{STATE_LABELS[state]} {t['confidence']:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 8), (x1 + label_size[0] + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Draw motion trail (past)
        if len(t["history"]) > 1:
            pts = [(int(p[0]), int(p[1])) for p in t["history"]]
            for i in range(1, len(pts)):
                alpha = i / len(pts)  # Fade older points
                trail_color = tuple(int(c * alpha) for c in color)
                cv2.line(frame, pts[i-1], pts[i], trail_color, 2)

        # Draw future trajectory (prediction)
        future_pts = t.get("future_trajectory", [])
        # Only draw future prediction when actively tracking
        if future_pts and state == ACTIVE:
            for i in range(0, len(future_pts), 3):  # Every 3rd frame for a dotted line effect
                pt = (int(future_pts[i][0]), int(future_pts[i][1]))
                # Fade out the prediction as it goes further into the future
                alpha = 1.0 - (i / len(future_pts))
                future_color = tuple(int(c * alpha) for c in color)
                cv2.circle(frame, pt, 2, future_color, -1)

    # Drawing the top bar
    cv2.rectangle(frame, (0, 0), (w, 35), (30, 30, 30), -1)
    hud_text = f"IRDDT | Frame: {frame_num} | Tracks: {len(tracks)} | FPS: {fps_val:.1f}"
    cv2.putText(frame, hud_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Drawing the bottom bar
    active = sum(1 for t in tracks if t["state"] == ACTIVE)
    lost = sum(1 for t in tracks if t["state"] == LOST)
    absent = sum(1 for t in tracks if t["state"] == ABSENT)

    cv2.rectangle(frame, (0, h - 30), (w, h), (30, 30, 30), -1)
    status = f"ACTIVE: {active}  LOST: {lost}  ABSENT: {absent}"
    cv2.putText(frame, status, (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    return frame


def load_ground_truth(seq_dir):
    # loading ground truth from IR_label.json for comparison.
    import json
    label_path = os.path.join(seq_dir, "IR_label.json")
   
    with open(label_path) as f:
        data = json.load(f)
    return data["exist"], data["gt_rect"]


def run_sequence(model_path, dataset_dir, split, seq_name, output_dir, conf_thresh=0.25):
    # running the full detection + tracking pipeline on one set of frames (video).

    # finding the sequence.
    seq_dir = os.path.join(dataset_dir, split, seq_name)

    
    # getting frame files in a sorted manner, to maintain order of video.
    frames = []
    for f in os.listdir(seq_dir):
        if f.endswith(".jpg"): #because we also have a .json file.
            frames.append(f)
    frames = sorted(frames)
    print(f"Frames:   {len(frames)}")


    # loading YOLO model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # Initializing "tracker" object from tracker.py and auto calling the __init__ function of the "Tracker" class.
    # Tracker class is resposible for maintaining all the individual tracks in the video.
    tracker = Tracker(
        high_thresh=0.5, # Only trust YOLO completely if it is at least 50% sure it found a drone (First Pass). 
        low_thresh=0.25, # If YOLO is between 25%-50% sure, keep it in mind (it might be a blurry drone), but ignore anything below 10% (Second Pass).
        iou_thresh=0.15, # The minimum overlap required to consider two boxes a match.
        dist_thresh=100.0, # If the boxes don't overlap (IoU fails) and center of the new YOLO box is within 100 pixels of our tracker's expected center, we accept the match.
        max_lost=45, # If we lose sight of the drone for 45 frames (about 1.5 secs), mark it as "LOST"and draw the boxes predicted by our kalman filter meanwhile.
        max_absent=45, # If 45 frames pass, the tracker moves to an "ABSENT" state. It gives the drone another 45 frames to reappear (total 3 seconds in the LOST+ABSENT state). the tracker then deletes its memory of that drone forever.
        min_hits_to_persist=30, # Don't start tracking something unless YOLO has consistently seen it for at least 30 frames, to avoid false Positives.
    )

    # loading ground truth, so we can visually compare GT vs our code's detection and tracking accuracy.
    gt_exist, gt_rect = load_ground_truth(seq_dir)

    # Setting up video writer
    os.makedirs(output_dir, exist_ok=True) # creating output folder.
    output_path = os.path.join(output_dir, f"{seq_name}_tracked.mp4") # output file name for the result video.
    first_frame = cv2.imread(os.path.join(seq_dir, frames[0])) # reading the first frame only.
    h, w = first_frame.shape[:2] # extracting the dimensions of the first original frame to create accurate output video size.
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h)) # this creates the actual output video frame by frame, we give it the frame size, frame rate to maintain and what video format to use.
    print(f"Processing...")


    total_time = 0 # stopwatch for FPS Calulation.

    for i, fname in enumerate(frames):
        frame = cv2.imread(os.path.join(seq_dir, fname))
        t0 = time.time() # current time is stored.

        # YOLO detection on the current frame
        results = model.predict(frame, imgsz=640, conf=conf_thresh, verbose=False)

        # Storing in detections as [cx, cy, w, h, confidence], because our tracker needs center cordinates + width,height for tracking. 
        detections = [] # May store more than one detections like a 2D array.
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy() # YOLO returns top-left & bottom-right corners.
            conf = float(box.conf[0]) # We also need the Confidence value.
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            bw = x2 - x1
            bh = y2 - y1
            detections.append([cx, cy, bw, bh, conf]) 

        dets_array = np.array(detections) if detections else np.empty((0, 5)) # store it as an optimized numpy array, and return mpty list if no detections are found.

        # updating tracker
        tracks = tracker.update(dets_array) # Receives A list of dictionaries with keys: track_id, bbox, confidence, state, hits, history.

        dt = time.time() - t0
        total_time += dt
        fps = 1.0 / dt if dt > 0 else 0

        # drawing visualization
        frame = draw_tracks(frame, tracks, i, fps)

        # drawing ground truth if available
        if gt_exist is not None and i < len(gt_exist) and gt_exist[i] == 1:
            gx, gy, gw, gh = gt_rect[i]
            if gw > 0 and gh > 0:
                # Red color for Ground Truth: (0, 0, 255) in BGR
                cv2.rectangle(frame, (int(gx), int(gy)), (int(gx + gw), int(gy + gh)), (0, 0, 255), 1)
                cv2.putText(frame, "GT", (gx, gy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        writer.write(frame)

        # Progress
        if (i + 1) % 100 == 0 or i == len(frames) - 1:
            avg_fps = (i + 1) / total_time if total_time > 0 else 0
            print(f"  [{i+1}/{len(frames)}] avg FPS: {avg_fps:.1f}")

    writer.release()
    avg_fps = len(frames) / total_time if total_time > 0 else 0
    print(f"\nDone! Output saved to: {output_path}")
    print(f"Average FPS: {avg_fps:.1f}")


def main():
    parser = argparse.ArgumentParser(description="IRDDT Video Tracker")
    parser.add_argument("--split", default="test", help="folder name, in which to look for the sequence ") 
    parser.add_argument("--seq", required=True, help="folder name of the sequence to use for testing") 
    args = parser.parse_args()

    model_path = r"E:\Drone\IRDDT\best.pt"
    dataset_dir = r"E:\Drone\Anti-UAV410"
    output_dir = r"E:\Drone\IRDDT\outputs"
    conf_thresh = 0.25

    # Running the pipeline, passing the location for best YOLO model weight, Complete dataset Path,
    # Sub folder name(train, test, val), path for saving the output video, only detect a drone when YOLO is confidence > conf_thresh.
    run_sequence(model_path, dataset_dir, args.split, args.seq, output_dir, conf_thresh)


if __name__ == "__main__":
    main()
