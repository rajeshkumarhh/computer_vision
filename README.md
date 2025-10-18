
# Real-Time YOLOv8 Object Detection from Webcam 📹

This project demonstrates real-time object detection using the lightweight YOLOv8n (nano) model from the Ultralytics framework, processed via OpenCV with live camera input. It tracks and summarizes the objects detected during a session.

# 🌟 Features

Live Feed: Captures and processes video frames from the default webcam (cv2.VideoCapture(0)).

Real-Time Detection: Uses the pre-trained yolov8n.pt model for fast, on-the-fly inference.

Visualization: Displays bounding boxes, class labels, and confidence scores on the live video feed.

Session Summary: Prints a summary of the total unique objects detected and the most frequent detections after the session ends.

# 🛠️ Prerequisites

Ensure you have Python 3.x installed on your system. You will need a working webcam with the correct permissions.

# 📊 Example Output

A screenshot of the live detection:

Terminal Session Summary:

--- Session Summary ---
Total unique objects detected: 4
Most commonly detected objects:
- person: 580 times
- laptop: 580 times
- cell phone: 45 times
- cup: 12 times

--- All Detections ---
Detected: person with 0.95 confidence
Detected: laptop with 0.93 confidence
Detected: person with 0.96 confidence
Detected: cell phone with 0.81 confidence
...(and more)
# ⚙️ Project Structure

The repository primarily contains:

.
├── webcam_yolo.py     # The main object detection script.

├── requirements.txt   # List of all Python dependencies.

├── yolov8n.pt         # The pre-trained YOLOv8 nano model weights (downloaded on first run).

└── README.md          # This file.

# 🙏 Credits

Built using the Ultralytics YOLOv8 framework.

Utilizes OpenCV for video capture and display.
