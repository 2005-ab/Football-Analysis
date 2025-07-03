import os
import cv2
import pickle
import torch
import numpy as np
from ultralytics import YOLO
import supervision as sv
from supervision.detection.core import Detections
from collections import defaultdict
from team_assigner.team_assigner import TeamAssigner


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.track_history = defaultdict(list)  # Track history for each team
        self.max_track_age = 30  # Maximum frames to keep a track alive when lost
        self.track_age = defaultdict(int)  # Age counter for each track
        self.next_id = 1  # For assigning new IDs when needed
        
        # Team tracking parameters
        self.team1_tracks = {}  # Track players from team 1
        self.team2_tracks = {}  # Track players from team 2
        
        # Use the existing TeamAssigner for robust team assignment
        self.team_assigner = TeamAssigner()
        
        # Tracking quality metrics
        self.tracking_metrics = {
            'total_tracks': 0,
            'stable_tracks': 0,  # tracks that last more than 10 frames
            'average_track_length': 0,
            'track_switches': 0,  # when a detection switches to a different track
            'detection_confidence_avg': 0,
            'frame_coverage': 0,  # percentage of frames with detections
            'team1_players_avg': 0,
            'team2_players_avg': 0
        }
        # Load pitch detection model
        self.pitch_model = YOLO('models/best_pitch.pt')
        self.pitch_polygon = None

    def detect_frames(self, frames, conf=0.2, batch_size=16):
        dets = []
        for i in range(0, len(frames), batch_size):
            dets_batch = self.model.predict(frames[i:i + batch_size], conf=conf)
            dets.extend(dets_batch)
        return dets

    def extract_jersey_color(self, frame, bbox):
        """Extract dominant color from player jersey area with enhanced accuracy"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure valid bbox
            if x2 <= x1 or y2 <= y1:
                return None
            
            # Focus on upper body area (jersey) - more precise region
            height = y2 - y1
            width = x2 - x1
            
            # Jersey region: upper 60% of the body, excluding very top (head) and bottom (shorts)
            jersey_y1 = y1 + int(height * 0.15)  # Start from 15% down (below head)
            jersey_y2 = y1 + int(height * 0.75)  # End at 75% down (above shorts)
            
            # Also focus on center of jersey to avoid arm colors
            jersey_x1 = x1 + int(width * 0.2)   # 20% from left edge
            jersey_x2 = x2 - int(width * 0.2)   # 20% from right edge
            
            crop = frame[jersey_y1:jersey_y2, jersey_x1:jersey_x2]
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                return None
            
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            
            # Create mask to exclude very dark and very bright areas (shadows, highlights)
            v_channel = hsv[:, :, 2]  # Value channel
            s_channel = hsv[:, :, 1]  # Saturation channel
            
            # Create mask for valid color regions
            valid_mask = (v_channel > 30) & (v_channel < 220) & (s_channel > 30)
            
            if not np.any(valid_mask):
                return None
            
            # Apply mask to get only valid color regions
            valid_hsv = hsv[valid_mask]
            
            if len(valid_hsv) == 0:
                return None
            
            # Calculate histogram for hue channel with more bins for precision
            hist = cv2.calcHist([valid_hsv], [0], None, [180], [0, 180])
            
            # Find the dominant hue (excluding very low counts)
            min_count = np.max(hist) * 0.1  # At least 10% of max count
            dominant_indices = np.where(hist >= min_count)[0]
            
            if len(dominant_indices) == 0:
                return None
            
            # Use the most frequent hue
            dominant_hue = dominant_indices[np.argmax(hist[dominant_indices])]
            
            # Get average saturation and value for this hue
            hue_mask = valid_hsv[:, 0] == dominant_hue
            if np.any(hue_mask):
                avg_s = np.mean(valid_hsv[hue_mask, 1])
                avg_v = np.mean(valid_hsv[hue_mask, 2])
            else:
                avg_s = 128
                avg_v = 128
            
            # Convert back to BGR
            dominant_color = cv2.cvtColor(
                np.uint8([[[dominant_hue, avg_s, avg_v]]]), 
                cv2.COLOR_HSV2BGR
            )[0][0]
            
            return dominant_color
            
        except Exception as e:
            print(f"Color extraction error: {e}")
            return None

    def assign_team(self, frame, bbox, track_id):
        """Assign player to team using the TeamAssigner"""
        return self.team_assigner.get_player_team(frame, bbox, track_id)

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

    def match_tracks(self, current_detections, previous_tracks):
        """Match current detections to previous tracks using IoU"""
        if not previous_tracks:
            return {}, list(range(len(current_detections)))
        
        matches = {}
        unmatched_current = list(range(len(current_detections)))
        unmatched_previous = list(previous_tracks.keys())
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(current_detections), len(previous_tracks)))
        for i, det in enumerate(current_detections):
            for j, (prev_id, prev_info) in enumerate(previous_tracks.items()):
                iou_matrix[i, j] = self.calculate_iou(det['bbox'], prev_info['bbox'])
        
        # Greedy matching with IoU threshold
        iou_threshold = 0.3
        while True:
            if len(unmatched_current) == 0 or len(unmatched_previous) == 0:
                break
                
            max_iou = 0
            best_match = None
            
            for i in unmatched_current:
                for j_idx, j in enumerate(unmatched_previous):
                    if iou_matrix[i, j_idx] > max_iou and iou_matrix[i, j_idx] > iou_threshold:
                        max_iou = iou_matrix[i, j_idx]
                        best_match = (i, j)
            
            if best_match is None:
                break
                
            curr_idx, prev_id = best_match
            
            matches[curr_idx] = prev_id
            unmatched_current.remove(curr_idx)
            unmatched_previous.remove(prev_id)
        
        return matches, unmatched_current

    def is_inside_pitch(self, center, pitch_polygon):
        polygon_np = np.array(pitch_polygon, dtype=np.int32)
        return cv2.pointPolygonTest(polygon_np, center, False) >= 0

    def get_object_tracks(self, frames, stub_path=None):
        if stub_path and os.path.exists(stub_path):
            return pickle.load(open(stub_path, 'rb'))

        dets = self.detect_frames(frames)
        tracks = {"team1_players": [], "team2_players": [], "ball": []}

        # --- PITCH DETECTION ---
        if self.pitch_polygon is None:
            pitch_results = self.pitch_model(frames[0])
            pitch_boxes = pitch_results[0].boxes.xyxy.cpu().numpy()  # shape: (N, 4)
            if len(pitch_boxes) == 0:
                raise ValueError("No pitch detected in the first frame!")
            # Union of all boxes
            x1 = int(np.min(pitch_boxes[:, 0]))
            y1 = int(np.min(pitch_boxes[:, 1]))
            x2 = int(np.max(pitch_boxes[:, 2]))
            y2 = int(np.max(pitch_boxes[:, 3]))
            self.pitch_polygon = [
                (x1, y1),
                (x2, y1),
                (x2, y2),
                (x1, y2)
            ]
            print("[DEBUG] Pitch polygon (union of all boxes):", self.pitch_polygon)

        # Reset tracking state
        self.track_history.clear()
        self.track_age.clear()
        self.next_id = 1
        self.team1_tracks.clear()
        self.team2_tracks.clear()

        total_detections = 0
        total_confidence = 0
        frames_with_detections = 0
        team1_counts = []
        team2_counts = []
        clustering_initialized = False

        for idx, det in enumerate(dets):
            boxes = det.boxes
            names = det.names
            inv = {v: k for k, v in names.items()}

            if boxes is None or len(boxes) == 0:
                tracks["team1_players"].append(self.team1_tracks.copy())
                tracks["team2_players"].append(self.team2_tracks.copy())
                tracks["ball"].append({})
                team1_counts.append(len(self.team1_tracks))
                team2_counts.append(len(self.team2_tracks))
                continue

            valid_indices = []
            for i, box in enumerate(boxes):
                cls_id = box.cls
                cls_id_int = inv.get(cls_id, -1) if isinstance(cls_id, str) else int(cls_id)
                cls_name = names[cls_id_int].lower()
                if cls_name in ["player", "ball"]:
                    valid_indices.append(i)

            if not valid_indices:
                tracks["team1_players"].append(self.team1_tracks.copy())
                tracks["team2_players"].append(self.team2_tracks.copy())
                tracks["ball"].append({})
                team1_counts.append(len(self.team1_tracks))
                team2_counts.append(len(self.team2_tracks))
                continue

            boxes = boxes[valid_indices]
            frames_with_detections += 1

            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            cls_fixed = [inv.get(c, -1) if isinstance(c, str) else int(c) for c in cls]
            cls = np.array(cls_fixed)
            sup = Detections(xyxy=xyxy, confidence=conf, class_id=cls)
            with_tracks = self.tracker.update_with_detections(sup)

            current_ball = None
            current_team1_players = {}
            current_team2_players = {}
            filtered_player_detections_for_clustering = {}

            for det in with_tracks:
                bbox = det[0].tolist()
                class_id = det[3]
                track_id = int(det[4])
                class_id_int = inv.get(class_id, -1) if isinstance(class_id, str) else int(class_id)
                cls = names[class_id_int].lower()

                if cls == "player":
                    cx = int((bbox[0] + bbox[2]) / 2)
                    cy = int((bbox[1] + bbox[3]) / 2)
                    if self.is_inside_pitch((cx, cy), self.pitch_polygon):
                        player_info = {
                            'bbox': bbox,
                            'confidence': det[2],
                            'team': None
                        }
                        filtered_player_detections_for_clustering[track_id] = player_info
                        total_detections += 1
                        total_confidence += det[2]
                elif cls == "ball":
                    current_ball = {'bbox': bbox}

            # Team color clustering
            if not clustering_initialized and len(filtered_player_detections_for_clustering) >= 4:
                self.team_assigner.assign_team_color(frames[idx], filtered_player_detections_for_clustering)
                clustering_initialized = True
                print("[DEBUG] Team color clustering initialized.")
                print("[DEBUG] Team 1 color (BGR):", self.team_assigner.team_colors.get(1))
                print("[DEBUG] Team 2 color (BGR):", self.team_assigner.team_colors.get(2))

            # Assign teams - only accept the two team colors, ignore others
            for track_id, player_info in filtered_player_detections_for_clustering.items():
                if clustering_initialized:
                    jersey_color = self.team_assigner.get_player_color(frames[idx], player_info['bbox'])
                    player_info['jersey_color'] = jersey_color
                    team1_color = self.team_assigner.team_colors.get(1)
                    team2_color = self.team_assigner.team_colors.get(2)
                    if jersey_color is not None and team1_color is not None and team2_color is not None:
                        dist1 = np.linalg.norm(jersey_color - team1_color)
                        dist2 = np.linalg.norm(jersey_color - team2_color)
                        color_threshold = 50
                        if dist1 < color_threshold and dist1 < dist2:
                            team = 1
                        elif dist2 < color_threshold and dist2 < dist1:
                            team = 2
                        else:
                            team = None
                    else:
                        team = None
                else:
                    team = None
                    player_info['jersey_color'] = None
                player_info['team'] = team
                if team == 1:
                    current_team1_players[track_id] = player_info
                elif team == 2:
                    current_team2_players[track_id] = player_info

            self.team1_tracks = current_team1_players
            self.team2_tracks = current_team2_players
            tracks["team1_players"].append(self.team1_tracks.copy())
            tracks["team2_players"].append(self.team2_tracks.copy())
            tracks["ball"].append({1: current_ball} if current_ball else {})
            team1_counts.append(len(self.team1_tracks))
            team2_counts.append(len(self.team2_tracks))

        self.tracking_metrics['total_tracks'] = total_detections
        self.tracking_metrics['frame_coverage'] = frames_with_detections / len(frames) * 100
        self.tracking_metrics['team1_players_avg'] = np.mean(team1_counts) if team1_counts else 0
        self.tracking_metrics['team2_players_avg'] = np.mean(team2_counts) if team2_counts else 0
        if total_detections > 0:
            self.tracking_metrics['detection_confidence_avg'] = total_confidence / total_detections
        return tracks

    def get_tracking_quality_report(self):
        """Generate a comprehensive tracking quality report"""
        report = f"""
🔍 TEAM TRACKING QUALITY REPORT
{'='*50}
📊 Basic Metrics:
   • Total detections: {self.tracking_metrics['total_tracks']}
   • Frame coverage: {self.tracking_metrics['frame_coverage']:.1f}%
   • Average detection confidence: {self.tracking_metrics['detection_confidence_avg']:.3f}
   • Team 1 players (avg): {self.tracking_metrics['team1_players_avg']:.1f}
   • Team 2 players (avg): {self.tracking_metrics['team2_players_avg']:.1f}

📈 Team Balance:
   • Total players per frame: {self.tracking_metrics['team1_players_avg'] + self.tracking_metrics['team2_players_avg']:.1f}
   • Team 1 ratio: {self.tracking_metrics['team1_players_avg']/(self.tracking_metrics['team1_players_avg'] + self.tracking_metrics['team2_players_avg'])*100:.1f}%
   • Team 2 ratio: {self.tracking_metrics['team2_players_avg']/(self.tracking_metrics['team1_players_avg'] + self.tracking_metrics['team2_players_avg'])*100:.1f}%

🎯 Quality Indicators:
   • Coverage quality: {'✅ Good' if self.tracking_metrics['frame_coverage'] > 80 else '⚠️ Needs improvement'}
   • Confidence quality: {'✅ Good' if self.tracking_metrics['detection_confidence_avg'] > 0.5 else '⚠️ Low confidence'}
   • Team balance: {'✅ Balanced' if abs(self.tracking_metrics['team1_players_avg'] - self.tracking_metrics['team2_players_avg']) < 2 else '⚠️ Unbalanced'}

🎯 Recommendations:
"""
        
        if self.tracking_metrics['frame_coverage'] < 80:
            report += "   • Increase detection sensitivity (lower confidence threshold)\n"
        if self.tracking_metrics['detection_confidence_avg'] < 0.5:
            report += "   • Improve model training or use better model\n"
        if abs(self.tracking_metrics['team1_players_avg'] - self.tracking_metrics['team2_players_avg']) > 2:
            report += "   • Check team assignment logic - teams may be misclassified\n"
        
        report += "="*50
        return report

    def draw_ellipse(self, frame, bbox, color=(255, 255, 255), thickness=4):
        x1, y1, x2, y2 = map(int, bbox)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        axes = ((x2 - x1) // 2, (y2 - y1) // 2)
        cv2.ellipse(frame, center, axes, 0, 0, 360, color, thickness)

    def draw_triangle(self, frame, bbox, color=(0, 255, 255)):
        x1, y1, x2, y2 = map(int, bbox)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        size = min((x2 - x1), (y2 - y1)) // 4
        pts = np.array([[center[0], center[1] - size], [center[0] - size, center[1] + size], [center[0] + size, center[1] + size]], np.int32)
        cv2.fillPoly(frame, [pts], color)

    def draw_annotations(self, frames, tracks):
        """Draw team-based tracking annotations on frames with jersey colors"""
        annotated_frames = []
        
        for frame_idx, frame in enumerate(frames):
            img = frame.copy()
            
            # Draw team 1 players with jersey colors
            if frame_idx < len(tracks['team1_players']):
                for track_id, info in tracks['team1_players'][frame_idx].items():
                    bbox = info['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Use jersey color if available, otherwise default to red
                    if 'jersey_color' in info and info['jersey_color'] is not None:
                        color = tuple(int(x) for x in info['jersey_color'])
                    else:
                        color = (0, 0, 255)  # Red for team 1
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw team label
                    confidence_text = f"{info['confidence']:.2f}" if 'confidence' in info else "N/A"
                    cv2.putText(img, f"Team 1 ({confidence_text})", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw team 2 players with jersey colors
            if frame_idx < len(tracks['team2_players']):
                for track_id, info in tracks['team2_players'][frame_idx].items():
                    bbox = info['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Use jersey color if available, otherwise default to blue
                    if 'jersey_color' in info and info['jersey_color'] is not None:
                        color = tuple(int(x) for x in info['jersey_color'])
                    else:
                        color = (255, 0, 0)  # Blue for team 2
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw team label
                    confidence_text = f"{info['confidence']:.2f}" if 'confidence' in info else "N/A"
                    cv2.putText(img, f"Team 2 ({confidence_text})", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw ball
            if frame_idx < len(tracks['ball']) and tracks['ball'][frame_idx]:
                for _, ball_info in tracks['ball'][frame_idx].items():
                    bbox = ball_info['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(img, "BALL", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Add tracking quality overlay
            if frame_idx < len(tracks['team1_players']):
                team1_count = len(tracks['team1_players'][frame_idx])
                team2_count = len(tracks['team2_players'][frame_idx])
                total_players = team1_count + team2_count
                
                cv2.putText(img, f"Team 1: {team1_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(img, f"Team 2: {team2_count}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(img, f"Total: {total_players}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(img, f"Frame: {frame_idx}", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            annotated_frames.append(img)
        
        return annotated_frames
