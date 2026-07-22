"""
Live webcam vehicle classification using the trained transfer-learning model
(transfer_best.keras — EfficientNetB0 backbone, 224x224 input, 11 classes).
 
This is a CLASSIFIER, not a detector: it labels the whole frame (or a chosen
region of it), it does not draw a bounding box around individual vehicles.
For actual box-drawing detection, use the separate YOLOv8 model instead
(see the note at the bottom of this file).
 
Setup:
    python -m venv venv
    source venv/bin/activate        # Windows: venv\\Scripts\\activate
    pip install tensorflow opencv-python numpy
 
Files needed in the same folder as this script:
    transfer_best.keras
    class_names.json        <- generate with export_class_names.py in Colab
 
Run:
    python camera_predict.py
    (press 'q' to quit)
"""
 
import json
import sys
import time
from pathlib import Path
 
import cv2
import numpy as np
import tensorflow as tf
 
# ---- Config ----
MODEL_PATH = "transfer_best.keras"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = 224          # must match training (Section 2 IMG_SIZE in the notebook)
CAMERA_INDEX = 0        # 0 = default webcam; try 1, 2... if you have multiple cameras
CONF_THRESHOLD = 0.5    # below this, label as "Uncertain" instead of trusting the top class
PREDICT_EVERY_N_FRAMES = 3  # skip frames between predictions to keep the feed smooth on CPU
 
 
def load_class_names(path):
    p = Path(path)
    if not p.is_file():
        sys.exit(
            f"ERROR: '{path}' not found.\n"
            "Run export_class_names.py in your Colab notebook (after train_generator "
            "exists) and put the downloaded class_names.json next to this script."
        )
    with open(p) as f:
        return json.load(f)
 
 
def preprocess_frame(frame_bgr):
    """Resize + convert BGR (OpenCV) -> RGB, and keep pixels in raw [0,255].
    IMPORTANT: do NOT divide by 255 here — EfficientNetB0 has its own built-in
    Rescaling/Normalization layer that expects raw [0,255] input (see the fix
    applied earlier in the notebook)."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE))
    x = resized.astype(np.float32)          # stays in [0, 255]
    x = np.expand_dims(x, axis=0)
    return x
 
 
def main():
    class_names = load_class_names(CLASS_NAMES_PATH)
    print(f"Loaded {len(class_names)} classes: {class_names}")
 
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded. Input shape:", model.input_shape)
 
    if model.output_shape[-1] != len(class_names):
        sys.exit(
            f"ERROR: model has {model.output_shape[-1]} output classes but "
            f"class_names.json has {len(class_names)} entries — these must match. "
            "Re-check you generated class_names.json from the same training run."
        )
 
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        sys.exit(f"ERROR: could not open camera index {CAMERA_INDEX}.")
 
    frame_count = 0
    last_label = "Warming up..."
    last_conf = 0.0
    fps_time = time.time()
 
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARNING: failed to grab frame from camera.")
            break
 
        frame_count += 1
        if frame_count % PREDICT_EVERY_N_FRAMES == 0:
            x = preprocess_frame(frame)
            probs = model.predict(x, verbose=0)[0]
            top_idx = int(np.argmax(probs))
            last_conf = float(probs[top_idx])
            last_label = class_names[top_idx] if last_conf >= CONF_THRESHOLD else "Uncertain"
 
        # ---- Overlay result on the frame ----
        display_frame = frame.copy()
        text = f"{last_label} ({last_conf * 100:.1f}%)"
        color = (0, 200, 0) if last_conf >= CONF_THRESHOLD else (0, 165, 255)
        cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(display_frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
 
        # FPS counter (approximate, since prediction runs every N frames)
        now = time.time()
        fps = 1.0 / (now - fps_time) if now != fps_time else 0.0
        fps_time = now
        cv2.putText(display_frame, f"{fps:.1f} FPS", (10, display_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
 
        cv2.imshow("Vehicle Classification (press q to quit)", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
 
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()
 
# ---------------------------------------------------------------------------
# Note: this classifies the WHOLE FRAME as one vehicle class — it assumes one
# vehicle roughly fills the camera view, and won't draw a box or handle
# multiple vehicles in frame. If you want that (box around each vehicle,
# works with cluttered/multi-vehicle scenes), use the YOLOv8 detection model
# from Section 16 of the notebook (yolov8_vehicle_detect/weights/best.pt)
# instead — Ultralytics supports a live webcam directly:
#
#   from ultralytics import YOLO
#   model = YOLO("best.pt")
#   model.predict(source=0, show=True)   # source=0 = webcam, draws boxes live
# ---------------------------------------------------------------------------