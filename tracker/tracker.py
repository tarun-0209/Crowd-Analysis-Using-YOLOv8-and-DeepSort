"""
Architecture:
1. Kalman filter predicts where each track should be
2. First pass: match HIGH confidence detections to tracks (IoU + Hungarian)
3. Second pass: match LOW confidence detections to remaining tracks (ByteTrack trick)
4. Unmatched detections → new tracks
5. Unmatched tracks → state machine (ACTIVE → LOST → ABSENT → deleted)
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from .kalman_filter import KalmanFilter

# Track States
ACTIVE = "ACTIVE"            # Drone locked, tracking normally
LOST = "LOST"                # Drone missed for a few frames, Kalman predicting
ABSENT = "ABSENT"            # Drone gone too long, search mode
RE_ACQUIRED = "RE_ACQUIRED"  # Drone found again after absence


# IoU Computation

def compute_iou_matrix(boxes_a, boxes_b):
    # Compute IoU between two sets of [cx, cy, w, h] boxes
    # Returns NxM matrix where result[i,j] = IoU(boxes_a[i], boxes_b[j])
    a = np.array(boxes_a)
    b = np.array(boxes_b)

    # Convert center format to corner format [x1, y1, x2, y2]
    a_x1, a_y1 = a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2
    a_x2, a_y2 = a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2
    b_x1, b_y1 = b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2
    b_x2, b_y2 = b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2

    iou_matrix = np.zeros((len(a), len(b)))
    for i in range(len(a)):
        xx1 = np.maximum(a_x1[i], b_x1)
        yy1 = np.maximum(a_y1[i], b_y1)
        xx2 = np.minimum(a_x2[i], b_x2)
        yy2 = np.minimum(a_y2[i], b_y2)

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_a = (a_x2[i] - a_x1[i]) * (a_y2[i] - a_y1[i])
        area_b = (b_x2 - b_x1) * (b_y2 - b_y1)
        union = area_a + area_b - inter

        iou_matrix[i] = np.where(union > 0, inter / union, 0)

    return iou_matrix


def compute_center_dist_matrix(boxes_a, boxes_b):
    # Compute Euclidean distance between centers of two sets of [cx, cy, w, h] boxes
    # Returns NxM matrix. Used as fallback when IoU fails during fast motion
    a = np.array(boxes_a)
    b = np.array(boxes_b)
    # Centers are already the first two elements
    dist_matrix = np.zeros((len(a), len(b)))
    for i in range(len(a)):
        dist_matrix[i] = np.sqrt((a[i, 0] - b[:, 0])**2 + (a[i, 1] - b[:, 1])**2)
    return dist_matrix


def compute_combined_cost(track_boxes, det_boxes, dist_thresh=100.0):
    # Combined cost matrix: uses IoU when boxes overlap, falls back to center distance (normalized) when they don't. This handles fast motion where boxes may not overlap at all.
    iou = compute_iou_matrix(track_boxes, det_boxes)
    dist = compute_center_dist_matrix(track_boxes, det_boxes)

    # If the boxes overlap (IoU > 0), the cost is 1 - IoU => Costs range from 0.0 to 0.99.
    # If the boxes don't touch but the center distance is within the dist_thresh, the cost is calculated as: 0.5 + 0.5 * (Distance/Threshold) => Costs range from 0.5 to 1.0.
    # If they don't overlap and the distance is greater than the threshold, the cost is set to 1.5 => These are effectively ignored by the Hungarian algorithm.
    cost = np.zeros_like(iou)
    for i in range(len(track_boxes)):
        for j in range(len(det_boxes)):
            if iou[i, j] > 0:
                # Boxes overlap: use IoU cost (lower is better)
                cost[i, j] = 1 - iou[i, j]
            elif dist[i, j] < dist_thresh:
                # No overlap but close enough: use distance cost (0.5 to 1.0 range)
                cost[i, j] = 0.5 + 0.5 * (dist[i, j] / dist_thresh)
            else:
                # Too far away: impossible match
                cost[i, j] = 1.5

    return cost, iou, dist


# Track Class

class Track:
    # Single object track with Kalman filter and state machine
    _next_id = 1

    def __init__(self, bbox, confidence):
        self.id = Track._next_id
        Track._next_id += 1
        self.kf = KalmanFilter(bbox)
        self.confidence = confidence
        self.state = ACTIVE
        self.hits = 1                # Times matched to a detection
        self.time_since_update = 0   # Frames since last match
        self.history = [bbox.copy()] # Trail for visualization
        self.reacquire_hit_count = 0 # For visualizing the reacquire state

    def predict(self):
        # Predict next position. Called once per frame.
        self.kf.predict()
        self.time_since_update += 1
        # return self.kf.get_state()

    def update(self, bbox, confidence):
        # Update with a matched detection
        self.kf.update(bbox)
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0
        self.history.append(self.kf.get_state().copy())
        if len(self.history) > 30:
            self.history.pop(0)

        # State transitions on successful match
        if self.state in (LOST, ABSENT):
            self.state = RE_ACQUIRED
            self.reacquire_hit_count = self.hits 

        # Only transition to ACTIVE if we aren't currently "Re-acquiring"
        if self.state == RE_ACQUIRED:
            if (self.hits - self.reacquire_hit_count) > 15:
                self.state = ACTIVE
        else:
            self.state = ACTIVE

    def get_bbox(self):
        # Current bounding box [cx, cy, w, h]
        return self.kf.get_state()

    def get_future_trajectory(self, steps=45):
        # Get predicted future positions [cx, cy] for the next N frames
        return self.kf.predict_future(steps)


# Main Tracker
class Tracker:
    # Multi-object tracker inspired by ByteTrack.
    # Two-pass IoU association + Kalman filter + drone state management.
    
    def __init__(self, high_thresh=0.5, low_thresh=0.25, iou_thresh=0.15,
                 dist_thresh=100.0, max_lost=45, max_absent=45, min_hits_to_persist=30):
        
        # self.tracks is a list that holds all the current 'Track' objects (the drones being followed).
        self.tracks = []
        
        # Tracker configuration rules:
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.iou_thresh = iou_thresh
        self.dist_thresh = dist_thresh
        self.max_lost = max_lost
        self.max_absent = max_absent
        self.min_hits_to_persist = min_hits_to_persist

    def update(self, detections):
        # Process one frame of detections.
    
        # STEP 1: PREDICT (Project states forward)
        for track in self.tracks: # nothing happens here if it's the first frame.
            track.predict()

        # If YOLO found absolutely nothing in this frame, skip the matching math.
        if len(detections) == 0:
            # Tell all current tracks they were unmatched this frame.
            all_track_indexes = list(range(len(self.tracks))) # list of every active track's index.
            self._handle_unmatched(all_track_indexes)
            self._cleanup() # Remove any tracks that have been missing too long.
            return self._results()

        # STEP 2: SPLIT DETECTIONS BY CONFIDENCE
        # We separate YOLO's detections into two lists: "High Confidence" and "Low Confidence" (ByteTrack trick)
        high_dets = []
        low_dets = []
        
        # We also need to remember the original index number of each detection
        for i, det in enumerate(detections):
            confidence = det[4] 
            if confidence >= self.high_thresh:
                high_dets.append((i, det))
            elif confidence >= self.low_thresh:
                low_dets.append((i, det)) 
                
        # We create a list of all current track indices. 
        # As we match them to detections, we will remove them off this list giving priority to high-confidence detections first. (ByteTrack trick)
        unmatched_tracks = list(range(len(self.tracks))) # list of indexes of all the current tracks.
        
        # We keep a set of YOLO detection indices  that have been claimed.
        matched_det_set = set() # empty right now.

        # STEP 3: FIRST PASS (Match High Confidence detections)
        if len(high_dets) > 0 and len(unmatched_tracks) > 0: # if high_dets is not empty and we still have some existing tracks that we have not yet updated.
            
            track_boxes = []
            # Get the Kalman Filter's predicted position [cx, cy, w, h] for all the currently unmatched tracks.
            for track_index in unmatched_tracks:
                box = self.tracks[track_index].get_bbox()
                track_boxes.append(box)
                
            # Get the YOLO's predicted position for each drone from our High Confidence list.
            high_boxes = [det[1][:4] for det in high_dets]
            
            # The job is to build a mathematical grid that compares every single active track against every single new detection found in the current frame.
            # iou (Intersection over Union): A matrix containing the overlap percentage between every track and detection. A value of 1.0 means they are perfectly on top of each other; 0.0 means they don't touch at all.
            # dist (Euclidean Distance): A matrix containing the straight-line pixel distance between the center point of every track and the center point of every detection.
            # cost (The Assignment Score): This is the final "master score" derived from both IoU and Distance. It is designed such that a lower value represents a better match.
            cost, iou, dist = compute_combined_cost(track_boxes, high_boxes, self.dist_thresh) # we only need Cost
            
            # The Hungarian Algorithm. 
            # It looks at the cost matrix and proposes the best possible pairings of each track and drone.
            # row[i] contains the index of a track, col[i] contains the index of the detection assigned to that track.
            row, col = linear_sum_assignment(cost)

            # The Hungarian algorithm will always find a "best" match, even if the only available drone is 500 pixels away from the only available detection.
            matched_t = set()
            for r, c in zip(row, col): # zip = Parallel Iteration in two or more lists just to save memory.
                # Is the match actually good enough to accept?
                if iou[r, c] >= self.iou_thresh or dist[r, c] < self.dist_thresh: # This check prevents ID Jumping by checking a minimum threshold for a match to be accepted.
                    
                    # It must be a match.
                    track_index = unmatched_tracks[r]
                    original_det_index = high_dets[c][0]
                    det_box = high_dets[c][1][:4]
                    det_conf = high_dets[c][1][4]
                    
                    # Update the specific track's Kalman filter with the real YOLO box
                    self.tracks[track_index].update(det_box, det_conf)
                    
                    # Mark these as claimed.
                    matched_t.add(track_index)
                    matched_det_set.add(original_det_index)

            # Remove the successfully matched tracks from our unmatched list
            unmatched_tracks = [i for i in unmatched_tracks if i not in matched_t]


        # STEP 4: SECOND PASS (Match Low Confidence detections)
        # If we STILL have tracks left over, they might be blurry/occluded drones.
        # We repeat the exact same matching process, but using the `low_dets` list.
        if len(low_dets) > 0 and len(unmatched_tracks) > 0: # if low_dets is not empty and we still have some existing tracks that we have not yet updated.
            
            track_boxes = [self.tracks[i].get_bbox() for i in unmatched_tracks]
            low_boxes = [det[1][:4] for det in low_dets]
            
            cost, iou, dist = compute_combined_cost(track_boxes, low_boxes, self.dist_thresh)
            row, col = linear_sum_assignment(cost)

            rescued = set()
            for r, c in zip(row, col):
                if iou[r, c] >= self.iou_thresh or dist[r, c] < self.dist_thresh:
                    track_index = unmatched_tracks[r]
                    original_det_index = low_dets[c][0]
                    det_box = low_dets[c][1][:4]
                    det_conf = low_dets[c][1][4]
                    
                    self.tracks[track_index].update(det_box, det_conf)
                    rescued.add(track_index)
                    matched_det_set.add(original_det_index)

            unmatched_tracks = [i for i in unmatched_tracks if i not in rescued]


        # STEP 5: MARK LEFTOVER TRACKS AS LOST
        # Any track STILL on this list didn't find a YOLO box because it was not detected in this current frame. we just move them to LOST or ABSENT states.
        self._handle_unmatched(unmatched_tracks)

        # STEP 6: Re-acquiring LOST drones (via Gating) or Initializing New Tracks
        for i, det in enumerate(detections):
            # Only process High Confidence detections that weren't claimed in Steps 3 or 4
            if i not in matched_det_set and det[4] >= self.high_thresh:
                det_box = det[:4]      # The 4D YOLO box [x1, y1, x2, y2]
                det_conf = det[4]      # YOLO confidence score
                z = det_box            # Measurement vector
                
                best_m_dist = float('inf')
                best_track = None
                
                # 1. Filter for candidates that are currently out of sight
                reacquire_candidates = [t for t in self.tracks if t.state in ("LOST", "ABSENT")]
                
                for track in reacquire_candidates:
                    # 2. Calculate the Innovation (The Surprise),for every lost track, the system calculates the raw difference between the new YOLO box (z) and where the Kalman Filter predicted the drone would be (Hx).
                    # y = z - Hx
                    H = track.kf.observation_matrix
                    x = track.kf.state_vector
                    y = z - (H @ x)
                    
                    # 3. Calculate the Bubble Size (Innovation Covariance S). The system calculates the Innovation Covariance (S).
                    # This is the mathematical "size" of the search area, which combines the internal uncertainty of the track (P) and the noise of the camera (R).
                    # S = HPH.T + R
                    P = track.kf.state_uncertainty
                    R = track.kf.sensor_noise
                    S = (H @ P @ H.T) + R
                    
                    # 4. Calculate Mahalanobis Distance
                    # This measures how many "standard deviations" the box is from our prediction. 
                    # It scales the distance by the State Uncertainty (P). As the drone stays lost, the "doubt bubble" grows.
                    # Mahalanobis distance naturally shrinks the penalty for a large pixel gap as the doubt grows, allowing the tracker to stay flexible over time.
                    try:
                        S_inv = np.linalg.inv(S)
                        m_dist = np.sqrt(y.T @ S_inv @ y)
                    except np.linalg.LinAlgError:
                        continue # Skip if the math becomes unstable
                        
                    if m_dist < best_m_dist:
                        best_m_dist = m_dist
                        best_track = track

                # 5. THE GATE: 4.0 is a standard 99% confidence threshold
                if best_track is not None and best_m_dist < 4.0:
                    # 1. KILL TOXIC VELOCITY: Keep the filter object but zero out the 4 velocity slots
                    # [cx, cy, w, h, VX, VY, VW, VH] -> slots 4, 5, 6, 7
                    best_track.kf.state_vector[4:] = 0 
                    # 2. UPDATE: Use the existing KF (which has high uncertainty/high gain right now)
                    # This will "snap" the position immediately and begin learning the new speed.
                    best_track.update(det_box, det_conf)                  
                    matched_det_set.add(i)
                else:
                    # INITIALIZATION: Either too far from lost drones or no candidates
                    # This is officially a brand-new drone entering the frame
                    new_track = Track(det_box, det_conf)
                    self.tracks.append(new_track)
                    matched_det_set.add(i)
        
        # STEP 7: CLEANUP AND RETURN
        self._cleanup() # Throw away tracks marked as "DELETED"
        return self._results() # Hand the final data back to run_video.py

    def _handle_unmatched(self, indices):
        # Apply state machine transitions to unmatched tracks.
        for i in indices:
            t = self.tracks[i]
            
            if t.hits < self.min_hits_to_persist:
                t.state = "DELETED" # If YOLO only saw it for a few frames (noise), delete it immediately.
                continue
                
            if t.time_since_update <= self.max_lost:
                t.state = LOST # The drone is missing, but we still trust the Kalman Filter's prediction.
            elif t.time_since_update <= self.max_lost + self.max_absent:
                t.state = ABSENT # It's been gone too long; we keep the ID alive but stop being confident.
            else:
                t.state = "DELETED" # The drone hasn't reappeared within the time limit (3 seconds).

    def _cleanup(self):
        # Remove tracks marked for deletion.
        self.tracks = [t for t in self.tracks if t.state != "DELETED"]

    def _results(self):
        # Package current tracks into output format.
        return [
            {
                "track_id": t.id,
                "bbox": t.get_bbox(),         # [cx, cy, w, h]
                "confidence": t.confidence,
                "state": t.state,
                "hits": t.hits,
                "history": t.history,
                "future_trajectory": t.get_future_trajectory(45), # 45 frames = 1.5 seconds at 30FPS
            }
            for t in self.tracks if t.hits >= self.min_hits_to_persist
        ]

    def reset(self):
        # Clear all tracks.
        self.tracks = []
        Track._next_id = 1
