# Football Player Detection and Tracking

A comprehensive computer vision project for detecting and tracking football players in video footage using YOLO (You Only Look Once) models and advanced tracking algorithms.

## 🚀 Features

- **Player Detection**: Real-time football player detection using YOLOv8 models
- **Player Tracking**: Multi-object tracking with StrongSORT and other advanced trackers
- **Video Processing**: Batch processing of video clips with customizable parameters
- **Data Export**: CSV output of tracking data for analysis
- **Model Training**: Custom YOLO model training capabilities
- **Multiple Trackers**: Support for various tracking algorithms (StrongSORT, ByteTrack, etc.)

## 📁 Project Structure

```
strategy/
├── clips/                    # Input video clips
│   ├── city_build_up/       # Manchester City build-up play clips
│   ├── spurs_build_up/      # Tottenham Spurs build-up play clips
│   └── *_tracks/            # Tracking results and renders
├── csv_outputs/             # Exported tracking data
├── models/                  # YOLO model weights
├── renders/                 # Output videos with detections
├── trackers/                # Custom tracking implementations
├── training/                # Model training scripts and data
├── ultralytics/             # YOLOv8 framework
├── utils/                   # Utility functions
├── Yolov5_StrongSORT_OSNet/ # StrongSORT tracking implementation
├── clip_match.py           # Main video processing script
├── detect_players.py       # Player detection script
└── track_players.py        # Player tracking script
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd strategy
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLO models** (optional - will be downloaded automatically on first run)
   ```bash
   # Models will be automatically downloaded when needed
   ```

## 📖 Usage

### Basic Player Detection

```python
from detect_players import detect_players

# Detect players in a video
detect_players(
    source="clips/city_build_up/city_clip01.mp4",
    model="yolov8n.pt",
    conf=0.5,
    save=True
)
```

### Player Tracking

```python
from track_players import track_players

# Track players in a video
track_players(
    source="clips/city_build_up/city_clip01.mp4",
    tracker="strongsort",
    save_trajectories=True,
    save_video=True
)
```

### Batch Processing

```python
from clip_match import process_clips

# Process multiple clips
process_clips(
    clips_dir="clips/city_build_up/",
    output_dir="output_videos/",
    tracker="strongsort"
)
```

## 🔧 Configuration

### Model Options
- `yolov8n.pt` - Nano model (fastest, smallest)
- `yolov8s.pt` - Small model (balanced)
- `yolov8m.pt` - Medium model (good accuracy)
- `yolov8x.pt` - Extra large model (best accuracy)

### Tracker Options
- `strongsort` - StrongSORT with appearance features
- `bytetrack` - ByteTrack (fast, efficient)
- `botsort` - BoTSORT
- `ocsort` - OC-SORT

## 📊 Output

The system generates:
- **Detection videos**: Videos with bounding boxes around detected players
- **Tracking videos**: Videos with player trajectories and IDs
- **CSV files**: Detailed tracking data with coordinates, velocities, and IDs
- **Rendered clips**: Processed video clips with annotations

## 🎯 Use Cases

- **Tactical Analysis**: Analyze team formations and player movements
- **Performance Metrics**: Track player speed, distance covered, and positioning
- **Match Analysis**: Review key moments and player interactions
- **Training Data**: Generate labeled datasets for further ML training

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - YOLO implementation
- [StrongSORT](https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet) - Advanced tracking algorithm
- [OpenCV](https://opencv.org/) - Computer vision library

## 📞 Support

For questions and support, please open an issue on GitHub or contact the maintainers. 