# Real-Time Object Detection with YOLO and Python

A desktop object detection application built with **Python**, **YOLO**, **OpenCV**, **Tkinter**, and **Pandas**.

The application supports real-time webcam detection, image upload, bounding-box visualization, Persian object labels, and Excel export for detection logs.

## Features

- Real-time object detection from webcam
- Image upload and object detection
- Bounding boxes and confidence scores
- Persian translation for detected object labels
- Detection log export to Excel
- Simple desktop interface using Tkinter

## Technologies Used

- Python
- Ultralytics YOLO
- OpenCV
- Tkinter
- Pillow
- Pandas
- OpenPyXL

## Project Structure

```text
realtime-object-detection-yolo-python/
├── main.py
├── requirements.txt
├── .gitignore
├── README.md
└── sample_outputs/
    └── detections_sample.xlsx
```

## Installation

Clone the repository:

```bash
git clone https://github.com/shayanhd/realtime-object-detection-yolo-python.git
cd realtime-object-detection-yolo-python
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

```bash
python main.py
```

By default, the application uses `yolov8s.pt`. Ultralytics can download this model automatically on first run.

You can also set a custom model path:

Windows PowerShell:

```powershell
$env:YOLO_MODEL_PATH="yolov8n.pt"
python main.py
```

macOS / Linux:

```bash
YOLO_MODEL_PATH="yolov8n.pt" python main.py
```

## Important Note About Model Files

Large YOLO weight files such as `.pt` files are intentionally not included in this repository.

Ignored examples:

- `yolov8n.pt`
- `yolov8s.pt`
- `yolo11x.pt`

This keeps the repository lightweight and suitable for GitHub.

## Academic Context

This project was developed as an applied AI and computer vision project by **Shayan Hadian**, a Computer Engineering undergraduate student focused on Software Engineering, AI, and Data Science.

## Future Improvements

- Add screenshot previews
- Add object count statistics
- Add CSV export
- Add model selection inside the GUI
- Improve UI responsiveness

## Author

**Shayan Hadian**  
Computer Engineering Undergraduate  
GitHub: [shayanhd](https://github.com/shayanhd)
