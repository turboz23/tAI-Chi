"""
Tai Chi pose-detection test bench (MediaPipe Pose Landmarker).

Setup:
    pip install mediapipe opencv-python numpy

Controls (click the video window first):
    r : record the current pose as the reference pose (saved to reference_pose.json)
    q : quit

What it does:
    - Detects 33 body landmarks from the webcam
    - Computes 8 joint angles (elbows, shoulders, hips, knees) in 3D
    - Compares them to the reference pose and shows the worst-matching joint
    - Shows per-frame inference time so you can budget for embedded hardware
"""
import json
import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
MODEL_PATH = "pose_landmarker_lite.task"
REFERENCE_PATH = "reference_pose.json"

# Joint name -> (point A, vertex B, point C) using BlazePose landmark indices.
JOINTS = {
    "L_elbow": (11, 13, 15), "R_elbow": (12, 14, 16),
    "L_shoulder": (13, 11, 23), "R_shoulder": (14, 12, 24),
    "L_hip": (11, 23, 25), "R_hip": (12, 24, 26),
    "L_knee": (23, 25, 27), "R_knee": (24, 26, 28),
}
CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
    (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
]


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading pose model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def angle_deg(a, b, c):
    """Angle at vertex b formed by points a-b-c, in degrees."""
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def joint_angles(world_landmarks):
    """Compute all joint angles from 3D world landmarks (meters, hip-centered)."""
    pts = np.array([[l.x, l.y, l.z] for l in world_landmarks])
    return {name: angle_deg(pts[a], pts[b], pts[c]) for name, (a, b, c) in JOINTS.items()}


def draw_skeleton(frame, landmarks):
    h, w = frame.shape[:2]
    px = [(int(l.x * w), int(l.y * h)) for l in landmarks]
    for i, j in CONNECTIONS:
        cv2.line(frame, px[i], px[j], (0, 255, 0), 2)
    for idx in {i for pair in CONNECTIONS for i in pair}:
        cv2.circle(frame, px[idx], 4, (0, 0, 255), -1)


def put_text(frame, text, y, color=(255, 255, 255)):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)


def main():
    ensure_model()
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    reference = None
    if os.path.exists(REFERENCE_PATH):
        with open(REFERENCE_PATH) as f:
            reference = json.load(f)
        print(f"Loaded reference pose from {REFERENCE_PATH}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (try a different index in VideoCapture)")

    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror view, easier to follow
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts_ms = int((time.time() - t0) * 1000)
        t_start = time.perf_counter()
        result = landmarker.detect_for_video(image, ts_ms)
        infer_ms = (time.perf_counter() - t_start) * 1000

        put_text(frame, f"Inference: {infer_ms:.1f} ms", 25)

        if result.pose_landmarks and result.pose_world_landmarks:
            draw_skeleton(frame, result.pose_landmarks[0])
            angles = joint_angles(result.pose_world_landmarks[0])

            if reference:
                errors = {k: abs(angles[k] - reference[k]) for k in JOINTS}
                mean_err = sum(errors.values()) / len(errors)
                worst = max(errors, key=errors.get)
                color = (0, 255, 0) if mean_err < 15 else (0, 165, 255) if mean_err < 30 else (0, 0, 255)
                put_text(frame, f"Mean angle error: {mean_err:.1f} deg", 50, color)
                put_text(frame, f"Worst joint: {worst} ({errors[worst]:.0f} deg off)", 75, color)
            else:
                put_text(frame, "No reference yet - strike a pose and press 'r'", 50)
        else:
            put_text(frame, "No person detected", 50, (0, 0, 255))
            angles = None

        cv2.imshow("Tai Chi pose test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r") and angles:
            reference = angles
            with open(REFERENCE_PATH, "w") as f:
                json.dump(reference, f, indent=2)
            print(f"Saved reference pose to {REFERENCE_PATH}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
