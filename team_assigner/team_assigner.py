from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.track_team_history = defaultdict(list)  # Track team assignment history
        self.team_confidence = defaultdict(float)  # Confidence for each track's team assignment
        self.min_confidence_threshold = 0.6
        self.history_window = 10  # Number of frames to consider for team assignment
    
    def get_clustering_model(self, image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1, 3)

        # Check if we have enough distinct colors
        unique_colors = np.unique(image_2d, axis=0)
        if len(unique_colors) < 2:
            # Not enough distinct colors, return None
            return None

        # Perform K-means with 2 clusters
        try:
            kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1, random_state=42)
            kmeans.fit(image_2d)
            
            # Check if clustering actually found 2 distinct clusters
            if len(np.unique(kmeans.labels_)) < 2:
                return None
                
            return kmeans
        except Exception as e:
            print(f"Clustering error: {e}")
            return None

    def get_player_color(self, frame, bbox):
        try:
            image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            
            if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
                return np.array([128, 128, 128])  # Default gray color
            
            top_half_image = image[0:int(image.shape[0]/2), :]

            # Get Clustering model
            kmeans = self.get_clustering_model(top_half_image)
            
            if kmeans is None:
                # Fallback to median color if clustering fails
                return np.median(top_half_image.reshape(-1, 3), axis=0)

            # Get the cluster labels for each pixel
            labels = kmeans.labels_

            # Reshape the labels to the image shape
            clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

            # Get the player cluster
            corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], 
                             clustered_image[-1, 0], clustered_image[-1, -1]]
            non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
            player_cluster = 1 - non_player_cluster

            player_color = kmeans.cluster_centers_[player_cluster]
            return player_color
        except Exception as e:
            print(f"Error in get_player_color: {e}")
            return np.array([128, 128, 128])  # Default gray color

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        valid_detections = []
        
        for track_id, player_detection in player_detections.items():
            try:
                bbox = player_detection["bbox"]
                player_color = self.get_player_color(frame, bbox)
                player_colors.append(player_color)
                valid_detections.append(track_id)
            except Exception as e:
                print(f"Error processing player {track_id}: {e}")
                continue
        
        if len(player_colors) < 2:
            print("Warning: Not enough players for team clustering")
            return
        
        player_colors = np.array(player_colors)
        
        # Check if we have enough distinct colors for clustering
        unique_colors = np.unique(player_colors, axis=0)
        if len(unique_colors) < 2:
            print("Warning: Not enough distinct colors for team clustering")
            return
        
        # Perform K-means clustering
        try:
            kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
            kmeans.fit(player_colors)
            
            # Check if clustering actually found 2 distinct clusters
            if len(np.unique(kmeans.labels_)) < 2:
                print("Warning: Clustering found only one team, using fallback assignment")
                # Assign teams based on color similarity to first two colors
                first_color = player_colors[0]
                second_color = None
                for color in player_colors[1:]:
                    if np.linalg.norm(color - first_color) > 30:  # Threshold for color difference
                        second_color = color
                        break
                
                if second_color is None:
                    print("Warning: All colors too similar, cannot assign teams")
                    return
                
                # Simple assignment based on distance to first two colors
                labels = []
                for color in player_colors:
                    dist1 = np.linalg.norm(color - first_color)
                    dist2 = np.linalg.norm(color - second_color)
                    labels.append(0 if dist1 < dist2 else 1)
                
                # Create a simple clustering result
                class SimpleKMeans:
                    def __init__(self, labels, centers):
                        self.labels_ = np.array(labels)
                        self.cluster_centers_ = centers
                    
                    def predict(self, X):
                        predictions = []
                        for x in X:
                            dist1 = np.linalg.norm(x - self.cluster_centers_[0])
                            dist2 = np.linalg.norm(x - self.cluster_centers_[1])
                            predictions.append(0 if dist1 < dist2 else 1)
                        return np.array(predictions)
                
                kmeans = SimpleKMeans(labels, [first_color, second_color])
            else:
                print(f"Successfully clustered {len(player_colors)} players into 2 teams")

        except Exception as e:
            print(f"Clustering error: {e}")
            return

        self.kmeans = kmeans

        # Assign team colors based on clustering
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]
        
        # Initialize team assignments for valid detections
        for i, track_id in enumerate(valid_detections):
            team_id = kmeans.predict(player_colors[i].reshape(1, -1))[0] + 1
            self.player_team_dict[track_id] = team_id
            self.track_team_history[track_id] = [team_id]
            self.team_confidence[track_id] = 1.0

    def get_team_from_history(self, track_id):
        """Get team assignment based on recent history"""
        if track_id not in self.track_team_history:
            return None
        
        history = self.track_team_history[track_id]
        if len(history) == 0:
            return None
        
        # Return the most common team in recent history
        recent_history = history[-self.history_window:]
        team_counts = {}
        for team in recent_history:
            team_counts[team] = team_counts.get(team, 0) + 1
        
        if team_counts:
            return max(team_counts, key=team_counts.get)
        return None

    def update_team_confidence(self, track_id, current_team, color_distance):
        """Update confidence based on color consistency"""
        if track_id not in self.team_confidence:
            self.team_confidence[track_id] = 0.5
        
        # Calculate confidence based on color distance to team centers
        if current_team in self.team_colors:
            expected_color = self.team_colors[current_team]
            color_diff = np.linalg.norm(color_distance - expected_color)
            confidence = max(0.1, 1.0 - color_diff / 100.0)  # Normalize color difference
        else:
            confidence = 0.5
        
        # Smooth confidence update
        self.team_confidence[track_id] = 0.7 * self.team_confidence[track_id] + 0.3 * confidence

    def get_player_team(self, frame, player_bbox, track_id):
        # First check if we have a confident team assignment from history
        if track_id in self.player_team_dict:
            history_team = self.get_team_from_history(track_id)
            if history_team and self.team_confidence.get(track_id, 0) > self.min_confidence_threshold:
                return history_team
        
        # Get current player color
        try:
            player_color = self.get_player_color(frame, player_bbox)
        except:
            # If color extraction fails, return history team or None
            return self.get_team_from_history(track_id)
        
        # Predict team based on current color
        if hasattr(self, 'kmeans'):
            predicted_team = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1
            
            # Update confidence
            self.update_team_confidence(track_id, predicted_team, player_color)
            
            # Update history
            if track_id not in self.track_team_history:
                self.track_team_history[track_id] = []
            self.track_team_history[track_id].append(predicted_team)
            
            # Keep history window size
            if len(self.track_team_history[track_id]) > self.history_window:
                self.track_team_history[track_id] = self.track_team_history[track_id][-self.history_window:]
            
            # Update team dictionary
            self.player_team_dict[track_id] = predicted_team
            
            return predicted_team
        else:
            # No clustering model available, return history team
            return self.get_team_from_history(track_id)

    def get_team_confidence(self, track_id):
        """Get confidence level for a track's team assignment"""
        return self.team_confidence.get(track_id, 0.0)
