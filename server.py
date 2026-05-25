"""
Fall-detection inference server.

Loads a YOLO model once at startup and exposes:
    GET  /            HTML status page
    GET  /health      JSON health check (also confirms model is loaded)
    GET  /docs        FastAPI auto-generated API docs
    POST /detect      multipart image=<jpeg> -> JSON detection result

Detection logic:
    - If the model has a custom class with "fall" / "fallen" / "lying" in its
      name, the response uses that directly.
    - Otherwise it falls back to the COCO `person` class + a
      width-greater-than-height aspect-ratio heuristic.

Run:
    python server.py --model best_falling.pt --port 8000
    python server.py --model yolov8s.pt --port 8000     # generic + heuristic
"""

from __future__ import annotations

import argparse
import os
import threading
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from ultralytics import YOLO


_FALL_CLASS_HINTS = ("fall", "fallen", "lying", "down")

MODEL: YOLO | None = None
MODEL_NAME: str = ""
PERSON_CONF_THRESHOLD: float = 0.15
VERBOSE: bool = False
_INFER_LOCK = threading.Lock()

# Ultralytics' default tracker. "bytetrack.yaml" is faster + lighter than
# "botsort.yaml" and good enough for a single-camera, low-clutter setup.
TRACKER_CONFIG = "bytetrack.yaml"


def _is_fall_class(name: str) -> bool:
    """Match any class whose lowercased name contains a fall keyword.
    Handles e.g. 'Fall Detected', 'person_falling', 'lying_down'."""
    n = name.lower()
    return any(hint in n for hint in _FALL_CLASS_HINTS)


def detect_falls(jpeg_bytes: bytes) -> dict:
    if MODEL is None:
        raise RuntimeError("Model not loaded")

    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode JPEG")

    h, w = frame.shape[:2]

    t0 = time.perf_counter()
    with _INFER_LOCK:
        # `persist=True` keeps tracker state across calls so the same person
        # keeps the same ID frame-to-frame. Ultralytics' default tracker
        # (configured via TRACKER_CONFIG) handles ID assignment.
        results = MODEL.track(
            frame,
            persist=True,
            tracker=TRACKER_CONFIG,
            save=False,
            verbose=False,
        )
    infer_ms = (time.perf_counter() - t0) * 1000

    people: list[dict] = []
    falling = False
    all_seen: list[tuple[str, float]] = []  # for verbose logging

    for info in results:
        names = info.names
        for box in info.boxes:
            cls_id = int(box.cls[0])
            cls_name = names.get(cls_id, str(cls_id))
            conf = float(box.conf[0])
            all_seen.append((cls_name, conf))

            if conf < PERSON_CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            bw = x2 - x1
            bh = y2 - y1

            is_fall = False
            if _is_fall_class(cls_name):
                is_fall = True
            elif cls_name.lower() == "person":
                # Aspect-ratio heuristic: wider than tall ⇒ lying down.
                is_fall = bw > bh
            else:
                continue

            if is_fall:
                falling = True

            # box.id is a tensor when the tracker has assigned one, else None
            # (e.g. very first frames before the tracker initialises).
            track_id: int | None = None
            if getattr(box, "id", None) is not None:
                try:
                    track_id = int(box.id[0])
                except Exception:
                    track_id = None

            people.append({
                "class": cls_name,
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2],
                "is_falling": is_fall,
                "track_id": track_id,
            })

    if VERBOSE:
        if all_seen:
            seen_str = ", ".join(
                f"{n}={c:.2f}" for n, c in sorted(all_seen, key=lambda x: -x[1])[:6]
            )
        else:
            seen_str = "<nothing>"
        print(
            f"[detect] {w}x{h} infer={infer_ms:.0f}ms "
            f"detections={len(all_seen)} kept={len(people)} "
            f"falling={falling}  top: {seen_str}"
        )

    return {
        "falling": falling,
        "people": people,
        "infer_ms": round(infer_ms, 1),
        "frame_size": [w, h],
    }


app = FastAPI(title="Fall Detection Server")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return f"""<!DOCTYPE html>
<html><body style="font-family: monospace; padding: 2em; background: #111; color: #eee;">
<h1>Fall Detection Server</h1>
<p>Model: <code>{MODEL_NAME}</code></p>
<p>POST <code>/detect</code> with multipart <code>image=&lt;jpeg&gt;</code></p>
<p>GET <code>/health</code></p>
<p>Interactive docs: <a href="/docs" style="color:#9cf">/docs</a></p>
</body></html>"""


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if MODEL is not None else "model-not-loaded",
        "model": MODEL_NAME,
        "person_conf_threshold": PERSON_CONF_THRESHOLD,
    }


@app.post("/detect")
async def detect(image: UploadFile = File(...)) -> JSONResponse:
    try:
        body = await image.read()
        result = detect_falls(body)
        return JSONResponse(result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="best_falling.pt",
        help="Path to YOLO .pt model (default: best_falling.pt).",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--conf", type=float, default=0.15,
        help="Min confidence (matches app.py's 10%% baseline).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print every /detect call: classes seen, confidences, decision.",
    )
    args = parser.parse_args()

    global MODEL, MODEL_NAME, PERSON_CONF_THRESHOLD, VERBOSE
    PERSON_CONF_THRESHOLD = args.conf
    VERBOSE = args.verbose

    if not os.path.exists(args.model):
        raise SystemExit(f"Model not found: {args.model}")

    print(f"Loading {args.model}...")
    MODEL = YOLO(args.model)
    MODEL_NAME = os.path.basename(args.model)
    print(f"Model loaded. Classes: {MODEL.names}")
    print(f"Listening on http://{args.host}:{args.port}")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
