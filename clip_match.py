import os
import subprocess

# Full match video path
input_video = r"file:D:\strategy\full_match.mp4"  # Add 'file:' prefix to fix FFmpeg protocol issue

# Output base folders
output_base = r"D:\strategy\clips"
city_dir = os.path.join(output_base, "city_build_up")
spurs_dir = os.path.join(output_base, "spurs_build_up")

# Ensure directories exist
os.makedirs(city_dir, exist_ok=True)
os.makedirs(spurs_dir, exist_ok=True)

# Timestamp ranges
city_clips = [
    ("1:08", "1:29"),
    ("2:10", "2:40"),
    ("2:57", "3:54"),
    ("5:25", "6:30"),
    ("7:33", "8:11"),
    ("8:52", "9:30"),
    ("13:39", "14:19"),
    ("16:34", "16:47"),
    ("32:40", "33:20"),
]

spurs_clips = [
    ("1:39", "1:59"),
    ("4:11", "4:30"),
    ("9:38", "10:19"),
    ("12:55", "13:21"),
    ("1:03:34", "1:03:54"),
    ("1:08:12", "1:08:35"),
    ("24:26", "24:53"),
]

# FFmpeg clipping function
def create_clips(clips, out_folder, prefix):
    for idx, (start, end) in enumerate(clips, 1):
        output_path = os.path.join(out_folder, f"{prefix}_clip{idx:02}.mp4")
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", start,
            "-to", end,
            "-i", input_video,
            "-c:v", "libx264",
            "-crf", "20",
            "-preset", "fast",
            "-c:a", "aac",
            output_path
        ]
        print(f"▶️ Creating {output_path}")
        subprocess.run(cmd)

# Run extraction
create_clips(city_clips, city_dir, "city")
create_clips(spurs_clips, spurs_dir, "spurs")
