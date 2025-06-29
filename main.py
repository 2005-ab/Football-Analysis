from trackers.tracker import Tracker
from utils.video_utils import read_video, save_video

if __name__ == "__main__":
    video_path = "clips/spurs_build_up/spurs_clip06.mp4"
    model_path = "models/best3.pt"
    output_path = "output_videos/output_with_ellipses.avi"
    cache_path  = "runs/tracks_cache.pkl"

    # 1) load
    frames = read_video(video_path)

    # 2) detect & track
    tracker = Tracker(model_path)
    tracks = tracker.get_object_tracks(frames, stub_path=cache_path)

    # 3) draw ellipses
    annotated = tracker.draw_annotations(frames, tracks)

    # 4) save
    save_video(annotated, output_path, fps=24)
    print(f"✅ Saved annotated video to {output_path}")
