import cv2, os
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans

# —— CONFIG ——
INPUT_VIDEO   = 'clips/city_build_up/city_clip02.mp4'
MODEL_PATH    = 'models/best4.pt'
PITCH2D_DIR   = 'output_videos/2d_pitch_city_clip10'
NUM_FRAMES    = 100    # how many frames to process
BALL_CLASS    = 'ball' # name that your YOLO model uses for the ball
PLAYER_CLASS  = 'player'

# —— UTILITIES ——

def load_video_frames(path, num=None):
    cap, frames = cv2.VideoCapture(path), []
    while cap.isOpened() and len(frames) < (num or float('inf')):
        ret, f = cap.read()
        if not ret: break
        frames.append(f)
    cap.release(); return frames

def extract_dominant_color(crop):
    if crop.size==0:
        return np.zeros(3)
    small = cv2.resize(crop, (20,20), interpolation=cv2.INTER_AREA)
    pts   = small.reshape(-1,3)
    c    = KMeans(n_clusters=1, random_state=0).fit(pts).cluster_centers_[0]
    return c

def detect_players_and_ball(frames, model_path, conf=0.3, iou=0.5):
    model = YOLO(model_path)
    player_bboxes = []   # list of lists of [x1,y1,x2,y2]
    ball_centers  = []   # list of (cx,cy) or None
    for i,frm in enumerate(frames):
        res = model(frm, conf=conf, iou=iou, verbose=False)
        players, balls = [], []
        for r in res:
            if r.boxes is None: continue
            for box,cls in zip(r.boxes, r.boxes.cls.cpu().numpy()):
                name = r.names[int(cls)]
                xy   = box.xyxy.cpu().numpy().flatten().tolist()
                if name==PLAYER_CLASS:
                    players.append(xy)
                elif name==BALL_CLASS:
                    x1,y1,x2,y2 = map(int, xy)
                    balls.append(((x1+x2)//2, (y1+y2)//2))
        player_bboxes.append(players)
        # if multiple balls, choose the first
        ball_centers.append(balls[0] if balls else None)
        print(f"Frame {i}: {len(players)} players, ball @ {ball_centers[-1]}")
    return player_bboxes, ball_centers

def assign_teams(frames, player_bboxes):
    # gather all jersey colors
    colors = []
    for frm, dets in zip(frames, player_bboxes):
        for xy in dets:
            x1,y1,x2,y2 = map(int,xy)
            crop = cv2.cvtColor(frm[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            colors.append(extract_dominant_color(crop))
    if len(colors)<2:
        raise RuntimeError("Not enough players for clustering")
    km = KMeans(n_clusters=2, random_state=0).fit(np.vstack(colors))
    centers = km.cluster_centers_
    # per-frame assignment
    team_map = {}
    for f_idx, dets in enumerate(player_bboxes):
        n = len(dets)
        if n==0: continue
        # frame colors
        fc = []
        for xy in dets:
            x1,y1,x2,y2 = map(int,xy)
            crop = cv2.cvtColor(frames[f_idx][y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            fc.append(extract_dominant_color(crop))
        fc = np.vstack(fc)
        d = np.linalg.norm(fc[:,None,:] - centers[None,:,:], axis=2)
        lbl = np.argmin(d, axis=1)
        # enforce max-11
        for t in (0,1):
            idxs = np.where(lbl==t)[0]
            if len(idxs)>11:
                far = idxs[np.argsort(-d[idxs,t])[:len(idxs)-11]]
                lbl[far] = 1-t
        for p_idx, team in enumerate(lbl):
            team_map[(f_idx,p_idx)] = int(team)
        print(f" Frame {f_idx}: Team0={np.sum(lbl==0)} Team1={np.sum(lbl==1)}")
    return team_map

def save_2d_pitch_with_possession(frames, player_bboxes, ball_centers, team_map, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    H,W = 680, 1050
    team_colors = [(0,0,255),(255,0,0)]  # BGR
    for f_idx, (frm, dets, ball) in enumerate(zip(frames, player_bboxes, ball_centers)):
        canvas = np.ones((H,W,3),dtype=np.uint8)*255
        # pitch
        cv2.rectangle(canvas,(0,0),(W-1,H-1),(0,200,0),4)
        cv2.line(canvas,(W//2,0),(W//2,H),(0,200,0),2)
        cv2.circle(canvas,(W//2,H//2),90,(0,200,0),2)
        # plot players
        centers = []
        for p_idx, xy in enumerate(dets):
            x1,y1,x2,y2 = map(int,xy)
            cx,cy = (x1+x2)//2, (y1+y2)//2
            px,py = int(cx/frm.shape[1]*W), int(cy/frm.shape[0]*H)
            t      = team_map.get((f_idx,p_idx),0)
            cv2.circle(canvas,(px,py),12,team_colors[t],-1)
            centers.append((px,py,t))
        # plot ball
        possession = None
        if ball:
            bx,by = ball
            bpx,bpy = int(bx/frm.shape[1]*W), int(by/frm.shape[0]*H)
            cv2.circle(canvas,(bpx,bpy),8,(0,0,0),-1)
            # find nearest player center
            dists = [((px-bpx)**2+(py-bpy)**2, t) for px,py,t in centers]
            if dists:
                possession = min(dists, key=lambda x: x[0])[1]
        # header text
        text = "Possession: "
        if possession is None:
            text += "None"
            color = (0,0,0)
        else:
            text += f"Team {possession}"
            color = team_colors[possession]
        cv2.putText(canvas, text, (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
        # save
        cv2.imwrite(os.path.join(save_dir, f'frame_{f_idx:03d}.png'), canvas)
    print(f"Saved 2D+possession images → {save_dir}")

# —— MAIN ——
if __name__=='__main__':
    frames, player_bboxes = load_video_frames(INPUT_VIDEO, NUM_FRAMES), None
    print(f"Loaded {len(frames)} frames.")
    player_bboxes, ball_centers = detect_players_and_ball(frames, MODEL_PATH)
    team_map = assign_teams(frames, player_bboxes)
    save_2d_pitch_with_possession(frames, player_bboxes, ball_centers, team_map, PITCH2D_DIR)
