# Hand Gesture Brightness & Volume Control

`bright.py` lets you control your PC’s **brightness** and **volume** using hand gestures via webcam.

- **Right hand** → Brightness  
- **Left hand** → Volume  
- Smooth, real-time control using **MediaPipe**  
- On-screen bars show current brightness and volume

## Requirements

- Python 3.8+  
- OpenCV, MediaPipe, NumPy  
- screen_brightness_control, pycaw, comtypes  

Install dependencies:

```bash
pip install opencv-python mediapipe numpy screen-brightness-control pycaw comtypes
```

Usage

python bright.py

    Pinch with right hand → adjust brightness

    Pinch with left hand → adjust volume

    Press ESC to exit

Notes

    Works best on Windows

    Good lighting improves hand tracking
