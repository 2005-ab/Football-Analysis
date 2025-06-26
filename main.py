from utils import read_video, save_video
from trackers import Tracker

def main():
    video_frames = read_video(r'clips\city_build_up\city_clip06.mp4')  # ✅ raw string to fix \c

    tracker=Tracker('models/best.pt')

    tracks=tracker.get_object_tracks(video_frames)
    save_video(video_frames, 'output_videos/output_video.avi')         # ✅ valid path with extension

if __name__ == '__main__':
    main()
