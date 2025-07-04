import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import defaultdict
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import cv2
from team_assigner.team_assigner import TeamAssigner


class YOLOv8Tracker:
    def __init__(self, model_path, conf_threshold=0.3, iou_threshold=0.5):
        """
        Initialize YOLOv8 tracker with robust tracking capabilities
        
        Args:
            model_path: Path to YOLOv8 model (.pt file)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Tracking parameters
        self.next_track_id = 1
        self.tracks = {}  # {track_id: Track}
        self.track_history = defaultdict(list)
        self.max_disappeared = 30  # Max frames to keep track alive
        self.min_hits = 3  # Min detections before confirming track
        self.max_age = 50  # Max age for track
        
        # Appearance features for re-identification
        self.appearance_features = {}
        self.feature_history = defaultdict(list)
        
        self.team_assigner = TeamAssigner()
        
    def extract_appearance_feature(self, frame, bbox):
        """Extract simple color histogram as appearance feature"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            
            # Convert to HSV and compute histogram
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            return hist
        except:
            return None
    
    def calculate_similarity(self, feature1, feature2):
        """Calculate similarity between two appearance features"""
        if feature1 is None or feature2 is None:
            return 0.0
        try:
            return cosine_similarity(feature1.reshape(1, -1), feature2.reshape(1, -1))[0][0]
        except:
            return 0.0
    
    def calculate_iou(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def predict_tracks(self):
        """Predict next positions of existing tracks using simple motion model"""
        for track_id, track in self.tracks.items():
            if len(track['history']) >= 2:
                # Simple velocity prediction
                last_pos = track['history'][-1]
                prev_pos = track['history'][-2]
                
                # Calculate velocity
                vx = last_pos[0] - prev_pos[0]
                vy = last_pos[1] - prev_pos[1]
                
                # Predict next position
                predicted_x = last_pos[0] + vx
                predicted_y = last_pos[1] + vy
                
                track['predicted'] = (predicted_x, predicted_y)
    
    def match_detections_to_tracks(self, detections, frame):
        """Match detections to existing tracks using IoU and appearance"""
        if not self.tracks:
            return [], list(range(len(detections)))
        
        # Extract appearance features for new detections
        detection_features = []
        for det in detections:
            feature = self.extract_appearance_feature(frame, det['bbox'])
            detection_features.append(feature)
        
        # Calculate similarity matrix
        similarity_matrix = np.zeros((len(detections), len(self.tracks)))
        track_ids = list(self.tracks.keys())
        
        for i, det in enumerate(detections):
            for j, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                
                # IoU similarity
                if 'predicted' in track:
                    pred_bbox = self.center_to_bbox(track['predicted'], det['bbox'])
                    iou_sim = self.calculate_iou(det['bbox'], pred_bbox)
                else:
                    iou_sim = 0.0
                
                # Appearance similarity
                if track_id in self.appearance_features and detection_features[i] is not None:
                    app_sim = self.calculate_similarity(
                        self.appearance_features[track_id], 
                        detection_features[i]
                    )
                else:
                    app_sim = 0.0
                
                # Combined similarity (weighted)
                similarity_matrix[i, j] = 0.7 * iou_sim + 0.3 * app_sim
        
        # Hungarian algorithm for optimal matching
        from scipy.optimize import linear_sum_assignment
        cost_matrix = 1 - similarity_matrix  # Convert similarity to cost
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_detections = []
        unmatched_tracks = list(range(len(self.tracks)))
        
        for i, j in zip(row_indices, col_indices):
            if similarity_matrix[i, j] > 0.3:  # Minimum similarity threshold
                matches.append((i, track_ids[j]))
                unmatched_tracks.remove(j)
            else:
                unmatched_detections.append(i)
        
        # Add detections that weren't matched
        for i in range(len(detections)):
            if i not in [m[0] for m in matches]:
                unmatched_detections.append(i)
        
        return matches, unmatched_detections
    
    def center_to_bbox(self, center, reference_bbox):
        """Convert center point to bbox using reference bbox dimensions"""
        cx, cy = center
        x1, y1, x2, y2 = reference_bbox
        w, h = x2 - x1, y2 - y1
        return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
    
    def extract_jersey_color(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.array([0, 0, 0])
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = crop.reshape(-1, 3)
        median_color = np.median(crop, axis=0)
        return median_color
    
    def update_tracks(self, detections, frame):
        """Update tracks with new detections"""
        # Predict next positions
        self.predict_tracks()
        
        # Match detections to tracks
        matches, unmatched_detections = self.match_detections_to_tracks(detections, frame)
        
        # Update matched tracks
        for det_idx, track_id in matches:
            det = detections[det_idx]
            track = self.tracks[track_id]
            
            # Update track
            center = ((det['bbox'][0] + det['bbox'][2]) / 2, 
                     (det['bbox'][1] + det['bbox'][3]) / 2)
            track['history'].append(center)
            track['bbox'] = det['bbox']
            track['confidence'] = det['confidence']
            track['disappeared'] = 0
            track['hits'] += 1
            
            # Update appearance feature
            feature = self.extract_appearance_feature(frame, det['bbox'])
            if feature is not None:
                self.appearance_features[track_id] = feature
                self.feature_history[track_id].append(feature)
                # Keep only recent features
                if len(self.feature_history[track_id]) > 10:
                    self.feature_history[track_id] = self.feature_history[track_id][-10:]
            # Store jersey color on first confirmation
            if track['hits'] == self.min_hits and 'jersey_color' not in track:
                jersey_color = self.extract_jersey_color(frame, det['bbox'])
                track['jersey_color'] = jersey_color
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            center = ((det['bbox'][0] + det['bbox'][2]) / 2, 
                     (det['bbox'][1] + det['bbox'][3]) / 2)
            
            new_track = {
                'bbox': det['bbox'],
                'confidence': det['confidence'],
                'history': [center],
                'disappeared': 0,
                'hits': 1,
                'age': 0
            }
            
            self.tracks[self.next_track_id] = new_track
            
            # Extract appearance feature
            feature = self.extract_appearance_feature(frame, det['bbox'])
            if feature is not None:
                self.appearance_features[self.next_track_id] = feature
                self.feature_history[self.next_track_id] = [feature]
            
            self.next_track_id += 1
        
        # Update unmatched tracks
        for track_id in list(self.tracks.keys()):
            if track_id not in [m[1] for m in matches]:
                self.tracks[track_id]['disappeared'] += 1
                self.tracks[track_id]['age'] += 1
        
        # Remove old tracks
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if (track['disappeared'] > self.max_disappeared or 
                track['age'] > self.max_age or
                (track['hits'] < self.min_hits and track['disappeared'] > 10)):
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
            if track_id in self.appearance_features:
                del self.appearance_features[track_id]
            if track_id in self.feature_history:
                del self.feature_history[track_id]
    
    def assign_teams(self):
        # Gather jersey colors from confirmed tracks
        jersey_colors = []
        track_ids = []
        for track_id, track in self.tracks.items():
            if 'jersey_color' in track:
                jersey_colors.append(track['jersey_color'])
                track_ids.append(track_id)
        if len(jersey_colors) < 2:
            print('Not enough tracks for team clustering!')
            return {}
        jersey_colors = np.array(jersey_colors)
        kmeans = KMeans(n_clusters=2, random_state=42).fit(jersey_colors)
        labels = kmeans.labels_
        team_assignment = {track_ids[i]: int(labels[i]) for i in range(len(track_ids))}
        return team_assignment
    
    def detect_and_track(self, frames, stub_path=None):
        """Main detection and tracking pipeline"""
        if stub_path and os.path.exists(stub_path):
            print(f"Loading cached tracks from {stub_path}")
            return pickle.load(open(stub_path, 'rb'))
        
        tracks = {"players": [], "ball": []}
        
        print("Running YOLOv8 detection and tracking...")
        for frame_idx, frame in enumerate(frames):
            # Run YOLOv8 detection
            results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
            
            # Process detections
            detections = []
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        bbox = boxes.xyxy[i].cpu().numpy()
                        confidence = boxes.conf[i].cpu().numpy()
                        class_id = int(boxes.cls[i].cpu().numpy())
                        class_name = result.names[class_id]
                        
                        # Only track players and ball (exclude goalkeepers and referees)
                        if class_name in ['player', 'ball']:
                            detections.append({
                                'bbox': bbox.tolist(),
                                'confidence': float(confidence),
                                'class_name': class_name,
                                'class_id': class_id
                            })
            
            # Separate players and ball
            player_detections = [d for d in detections if d['class_name'] == 'player']
            ball_detections = [d for d in detections if d['class_name'] == 'ball']
            
            # Update tracks
            self.update_tracks(player_detections, frame)
            
            # Prepare output
            current_players = {}
            # Assign team colors using TeamAssigner
            self.team_assigner.assign_team_color(frame, self.tracks)
            for track_id, track in self.tracks.items():
                if track['hits'] >= self.min_hits:  # Only confirmed tracks
                    team = self.team_assigner.get_player_team(frame, track['bbox'], track_id)
                    current_players[track_id] = {
                        'bbox': track['bbox'],
                        'confidence': track['confidence'],
                        'age': track['age'],
                        'jersey_color': track.get('jersey_color', [0,0,0]),
                        'team': team if team is not None else 0
                    }
            
            # Handle ball tracking (simpler approach)
            current_ball = {}
            if ball_detections:
                best_ball = max(ball_detections, key=lambda x: x['confidence'])
                current_ball[1] = {'bbox': best_ball['bbox']}
            
            tracks["players"].append(current_players)
            tracks["ball"].append(current_ball)
            
            if frame_idx % 50 == 0:
                print(f"Processed {frame_idx}/{len(frames)} frames, {len(current_players)} active tracks")
        
        if stub_path:
            pickle.dump(tracks, open(stub_path, 'wb'))
            print(f"Saved tracks to {stub_path}")
        
        return tracks
    
    def draw_annotations(self, frames, tracks):
        """Draw tracking annotations on frames"""
        annotated_frames = []
        
        for frame_idx, frame in enumerate(frames):
            img = frame.copy()
            
            # Draw player tracks
            if frame_idx < len(tracks['players']):
                for track_id, info in tracks['players'][frame_idx].items():
                    bbox = info['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Draw bounding box
                    color = (0, 255, 0)  # Green for players
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw track ID
                    cv2.putText(img, f"ID:{track_id}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Draw team color if available
                    if 'jersey_color' in info:
                        team_color = tuple(int(x) for x in info['jersey_color'])
                        cv2.rectangle(img, (x1, y1), (x2, y2), team_color, 3)
            
            # Draw ball
            if frame_idx < len(tracks['ball']) and tracks['ball'][frame_idx]:
                for _, ball_info in tracks['ball'][frame_idx].items():
                    bbox = ball_info['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(img, "BALL", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            annotated_frames.append(img)
        
        return annotated_frames 