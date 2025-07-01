from trackers.tracker import Tracker
from utils.video_utils import read_video, save_video
from team_assigner.team_assigner import TeamAssigner
import numpy as np
from scipy.optimize import linear_sum_assignment

# ---------------------- helper functions ----------------------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# color distance metric
def color_distance(c1, c2):
    return np.linalg.norm(np.array(c1, float) - np.array(c2, float))

# ID restoration using IoU + color + Hungarian
def restore_player_ids_persistent(tracks,
                                  team_assigner: TeamAssigner,
                                  iou_threshold=0.3,
                                  color_threshold=50,
                                  max_missing=10,
                                  max_players=22):
    id_cache = {}
    next_id = 1
    new_tracks = []

    for frame_num, player_track in enumerate(tracks['players']):
        detections = []
        for raw_id, info in player_track.items():
            bbox = info['bbox']
            color = tuple(team_assigner.get_player_color(tracks['frames'][frame_num], bbox))
            detections.append({'raw_id': raw_id, 'bbox': bbox, 'color': color, 'info': info})

        cache_ids = list(id_cache.keys())
        N, M = len(detections), len(cache_ids)
        cost = np.zeros((N, M), float)

        # build cost matrix
        for i, det in enumerate(detections):
            for j, pid in enumerate(cache_ids):
                cinfo = id_cache[pid]
                spatial_cost = 1 - iou(det['bbox'], cinfo['bbox'])
                ap_cost = color_distance(det['color'], cinfo['color']) / 255.0
                cost[i, j] = spatial_cost + ap_cost

        matched_det, matched_pid = set(), set()
        if N and M:
            row_ind, col_ind = linear_sum_assignment(cost)
            for i, j in zip(row_ind, col_ind):
                if cost[i, j] < ((1 - iou_threshold) + (color_threshold / 255.0)):
                    matched_det.add(i)
                    matched_pid.add(cache_ids[j])

        frame_map = {}
        # matched
        for i in matched_det:
            pid = cache_ids[list(matched_det).index(i)]
            det = detections[i]
            frame_map[pid] = det['info']
            id_cache[pid] = {'bbox': det['bbox'], 'color': det['color'], 'last_seen': frame_num}

        # unmatched -> new IDs
        for i, det in enumerate(detections):
            if i not in matched_det and len(id_cache) < max_players:
                pid = next_id
                next_id += 1
                frame_map[pid] = det['info']
                id_cache[pid] = {'bbox': det['bbox'], 'color': det['color'], 'last_seen': frame_num}

        # expire old
        expired = [pid for pid, c in id_cache.items() if frame_num - c['last_seen'] > max_missing]
        for pid in expired:
            del id_cache[pid]

        new_tracks.append(frame_map)

    tracks['players'] = new_tracks
    return tracks

# renumber jersey slots within teams
def renumber_within_teams(tracks, max_per_team=11):
    last_map = {1: {}, 2: {}}

    for frame_idx, frame_players in enumerate(tracks['players']):
        new_map = {1: {}, 2: {}}
        by_team = {1: [], 2: []}
        for pid, info in frame_players.items():
            by_team.setdefault(info['team'], []).append(pid)

        for team, pids in by_team.items():
            used = set(last_map[team].values())
            free = [i for i in range(1, max_per_team+1) if i not in used]
            
            for pid in pids:
                if pid in last_map[team]:
                    new_map[team][pid] = last_map[team][pid]
                else:
                    # If no free jersey numbers, reuse the least recently used one
                    if not free:
                        # Find the least used jersey number or use 1 as fallback
                        jersey_counts = {}
                        for existing_pid, jersey_id in last_map[team].items():
                            jersey_counts[jersey_id] = jersey_counts.get(jersey_id, 0) + 1
                        
                        # Use jersey number 1 if no existing jerseys, otherwise use the least used one
                        if jersey_counts:
                            least_used_jersey = min(jersey_counts.keys(), key=lambda x: jersey_counts[x])
                            new_map[team][pid] = least_used_jersey
                        else:
                            new_map[team][pid] = 1
                    else:
                        new_map[team][pid] = free.pop(0)

        # write back
        last_map = {1: {}, 2: {}}
        for pid, info in frame_players.items():
            team = info['team']
            jid = new_map[team][pid]
            tracks['players'][frame_idx][pid]['jersey_id'] = jid
            last_map[team][pid] = jid

    return tracks

# Simple and stable ID restoration using only spatial overlap
def restore_player_ids_simple(tracks, iou_threshold=0.4, max_missing=5):
    """
    Simple ID restoration that prioritizes stability over optimal assignment
    """
    id_cache = {}  # {id: {'bbox': ..., 'last_seen': frame_num}}
    next_id = 1
    
    for frame_num, player_track in enumerate(tracks['players']):
        new_player_track = {}
        
        # First pass: try to match with existing cache IDs (prioritize keeping same ID)
        for raw_id, info in player_track.items():
            bbox = info['bbox']
            matched_id = None
            best_iou = 0
            
            # Try to match with any cached ID seen recently
            for cache_id, cache_info in id_cache.items():
                if frame_num - cache_info['last_seen'] > max_missing:
                    continue
                    
                overlap = iou(bbox, cache_info['bbox'])
                if overlap > iou_threshold and overlap > best_iou:
                    best_iou = overlap
                    matched_id = cache_id
            
            if matched_id is not None:
                # Use existing ID
                new_player_track[matched_id] = info
                id_cache[matched_id] = {'bbox': bbox, 'last_seen': frame_num}
            else:
                # Assign new ID
                new_player_track[next_id] = info
                id_cache[next_id] = {'bbox': bbox, 'last_seen': frame_num}
                next_id += 1
        
        # Clean up expired IDs
        expired_ids = [pid for pid, info in id_cache.items() 
                      if frame_num - info['last_seen'] > max_missing]
        for pid in expired_ids:
            del id_cache[pid]
        
        tracks['players'][frame_num] = new_player_track
    
    return tracks

# ---------------------- main flow ----------------------
if __name__ == "__main__":
    video_path = "clips/spurs_build_up/spurs_clip06.mp4"
    model_path = "models/best4.pt"
    output_path = "output_videos/output_with_annotations.avi"
    cache_path  = "runs/tracks_cache.pkl"

    # load and track
    frames = read_video(video_path)
    tracker = Tracker(model_path)
    tracks = tracker.get_object_tracks(frames, stub_path=cache_path)
    tracks['frames'] = frames

    # team assign
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frames[0], tracks['players'][0])

    # restore IDs + reassign team & jersey color
    tracks = restore_player_ids_simple(tracks)
    for f, player_track in enumerate(tracks['players']):
        for pid, tr in player_track.items():
            team = team_assigner.get_player_team(frames[f], tr['bbox'], pid)
            tracks['players'][f][pid].update({
                'team': team,
                'team_color': team_assigner.team_colors[team]
            })
            try:
                col = team_assigner.get_player_color(frames[f], tr['bbox'])
                col = tuple(int(x) for x in np.clip(col,0,255))
            except:
                col = (0,0,255)
            tracks['players'][f][pid]['jersey_color'] = col

    # renumber jerseys 1-11
    tracks = renumber_within_teams(tracks)

    # annotate & save
    annotated = tracker.draw_annotations(frames, tracks)
    save_video(annotated, output_path, fps=24)
    print(f"✅ Saved annotated video to {output_path}")