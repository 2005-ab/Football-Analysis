import cv2
import os
import numpy as np
import random
from ultralytics import YOLO
from sklearn.cluster import KMeans

# —— PATHS & PARAMS ——
INPUT_VIDEO   = 'clips/spurs_build_up/spurs_clip04.mp4'
MODEL_PATH    = 'models/best.pt'
PITCH2D_DIR   = 'output_videos/2d_pitch_spurs_clip04_01'
FPS           = 25      # for video writing if you need it
NUM_FRAMES    = 100     # number of frames to process

# —— UTILITIES ——

def load_video_frames(path, num_frames=None):
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < (num_frames or float('inf')):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def extract_dominant_color(crop):
    """Resize to small patch + 1‑cluster kmeans for robust jersey color."""
    if crop.size == 0:
        return np.array([0,0,0], dtype=float)
    small = cv2.resize(crop, (20,20), interpolation=cv2.INTER_AREA)
    data  = small.reshape(-1,3)
    kmeans= KMeans(n_clusters=1, random_state=0).fit(data)
    return kmeans.cluster_centers_[0]

def get_all_player_detections(frames, model_path, conf=0.3, iou=0.5):
    model = YOLO(model_path)
    all_det = []
    for i,frm in enumerate(frames):
        res = model(frm, conf=conf, iou=iou, verbose=False)
        dets = []
        for r in res:
            if r.boxes is None: continue
            for box in r.boxes:
                cls = int(box.cls.cpu().numpy())
                name= r.names[cls]
                if name!='player': continue
                xy  = box.xyxy.cpu().numpy().flatten()
                dets.append(xy.tolist())
        print(f'Frame {i}: {len(dets)} players')
        all_det.append(dets)
    return all_det

def assign_teams(frames, all_det):
    # 1) gather all colors
    colors = []
    for frm, dets in zip(frames, all_det):
        for xy in dets:
            x1,y1,x2,y2 = map(int,xy)
            crop = cv2.cvtColor(frm[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            colors.append(extract_dominant_color(crop))
    if len(colors)<2:
        raise RuntimeError("Not enough detections for clustering")
    colors = np.vstack(colors)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(colors)
    centroids = kmeans.cluster_centers_
    
    # 2) assign per-frame, enforce max-11
    team_map = {}
    ptr = 0
    for f_idx, dets in enumerate(all_det):
        n = len(dets)
        if n==0: 
            ptr += 0
            continue
        # extract frame colors again
        fc = []
        for xy in dets:
            x1,y1,x2,y2 = map(int,xy)
            crop = cv2.cvtColor(frames[f_idx][y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            fc.append(extract_dominant_color(crop))
        fc = np.vstack(fc)
        # compute distances to team centroids
        d = np.linalg.norm(fc[:,None,:] - centroids[None,:,:], axis=2)  # (n,2)
        lbl = np.argmin(d, axis=1)
        # enforce max 11
        for t in (0,1):
            idxs = np.where(lbl==t)[0]
            if len(idxs)>11:
                # farthest from its centroid get flipped
                dist_to_own = d[idxs, t]
                farthest = idxs[np.argsort(-dist_to_own)[:len(idxs)-11]]
                lbl[farthest] = 1-t
        for pd_idx, team in enumerate(lbl):
            team_map[(f_idx, pd_idx)] = int(team)
        ptr += n
        print(f' Frame {f_idx}: Team0={np.sum(lbl==0)}  Team1={np.sum(lbl==1)}')
    return team_map

def save_2d_pitch(frames, all_det, team_map, save_dir, pitch_size=(680,1050)):
    os.makedirs(save_dir, exist_ok=True)
    h2, w2 = pitch_size
    team_colors = [(0,0,255),(255,0,0)]  # BGR
    
    for f_idx, (frm, dets) in enumerate(zip(frames, all_det)):
        canvas = np.ones((h2,w2,3),dtype=np.uint8)*255
        # pitch lines
        cv2.rectangle(canvas, (0,0),(w2-1,h2-1),(0,200,0),4)
        cv2.line(canvas, (w2//2,0),(w2//2,h2),(0,200,0),2)
        cv2.circle(canvas, (w2//2,h2//2),90,(0,200,0),2)
        
        for p_idx, xy in enumerate(dets):
            x1,y1,x2,y2 = map(int,xy)
            cx, cy = (x1+x2)//2, (y1+y2)//2
            # normalize to pitch
            px = int(cx/frm.shape[1]*w2)
            py = int(cy/frm.shape[0]*h2)
            t   = team_map.get((f_idx,p_idx), 0)
            cv2.circle(canvas,(px,py),12, team_colors[t], -1)
        
        outp = os.path.join(save_dir, f'frame_{f_idx:03d}.png')
        cv2.imwrite(outp, canvas)
    print(f"Saved 2D pitch images → {save_dir}")

# —— MAIN ——
if __name__=='__main__':
    # 1) grab frames
    frames    = load_video_frames(INPUT_VIDEO, num_frames=NUM_FRAMES)
    print(f"Loaded {len(frames)} frames.")
    
    # 2) detect players
    all_det   = get_all_player_detections(frames, MODEL_PATH)
    
    # 3) global team clustering + per‑frame assignment
    team_map  = assign_teams(frames, all_det)
    
    # 4) draw & save 2D pitch images
    save_2d_pitch(frames, all_det, team_map, PITCH2D_DIR)
