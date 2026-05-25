# capstone-falldetection.src

YOLOv8 fall-detection inference server.

Runs on your laptop. The Raspbot car (in `capstone-sensors.src`) sends USB
camera frames over HTTP, this server runs YOLO on them and returns whether a
fall is detected. The car then reacts (stops, logs, shows on dashboard).

---

## What's in this repo

```text
capstone-falldetection.src/
├── server.py            # FastAPI server (NEW)
├── app.py               # original standalone webcam demo (kept for reference)
├── best_falling.pt      # custom-trained model (used by server.py)
├── yolov8s.pt           # generic COCO model (fallback)
├── classes.txt          # COCO class list (only used by app.py)
├── falling.mp4          # sample video for testing
├── requirements.txt
└── README.md
```

---

## Setup

```bash
cd capstone-falldetection.src

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

This pulls in PyTorch via `ultralytics`. First install is large
(~1 GB) and takes a few minutes.

---

## Run the server

```bash
source .venv/bin/activate
python server.py --model best_falling.pt --port 8000
```

Expected output:

```text
Loading best_falling.pt...
Model loaded. Classes: {0: 'fall', 1: 'no_fall'}   # or whatever your model has
Listening on http://0.0.0.0:8000
```

Useful flags:

```bash
python server.py --model yolov8s.pt          # generic model + aspect-ratio heuristic
python server.py --model best_falling.pt --conf 0.5   # higher confidence cutoff
python server.py --port 9000
```

---

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Status HTML page |
| GET | `/health` | `{"status":"ok","model":"..."}` |
| GET | `/docs` | Interactive Swagger UI |
| POST | `/detect` | multipart `image=<jpeg>` → JSON result |

### `/detect` response shape

```json
{
  "falling": true,
  "people": [
    {
      "class": "person",
      "confidence": 0.87,
      "bbox": [120, 80, 240, 200],
      "is_falling": true
    }
  ],
  "infer_ms": 42.3,
  "frame_size": [320, 240]
}
```

---

## Find your laptop's IP (for the car to reach)

macOS:

```bash
ipconfig getifaddr en0
```

Linux:

```bash
hostname -I
```

The Pi must be on the **same Wi-Fi / hotspot** as the laptop running this
server.

---

## Quick smoke test

In another terminal:

```bash
curl -s http://localhost:8000/health
curl -s -F image=@falling.mp4 http://localhost:8000/detect   # WON'T WORK on a video
```

For a single image:

```bash
ffmpeg -i falling.mp4 -frames:v 1 frame.jpg
curl -s -F image=@frame.jpg http://localhost:8000/detect | python -m json.tool
```

Or just open `http://localhost:8000/docs` and try `/detect` from the Swagger
UI.

---

## How the car uses it

See `capstone-sensors.src/README.md` → "Step 9: Run with fall detection".
The car runs a background thread that grabs USB-camera frames at ~5 FPS,
POSTs them to this endpoint, and stops the motors while `falling` is true.
