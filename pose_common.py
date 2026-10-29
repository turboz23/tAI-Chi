"""Shared helpers for the Tai Chi coach (used by extract_reference.py and coach.py)."""
import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task"
)

# Joint name -> (point A, vertex B, point C), BlazePose landmark indices.
# ORDER MATTERS: left/right pairs are adjacent so mirror_vector() can swap them.
JOINTS = {
    "L_elbow": (11, 13, 15), "R_elbow": (12, 14, 16),
    "L_shoulder": (13, 11, 23), "R_shoulder": (14, 12, 24),
    "L_hip": (11, 23, 25), "R_hip": (12, 24, 26),
    "L_knee": (23, 25, 27), "R_knee": (24, 26, 28),
}
JOINT_NAMES = list(JOINTS)
CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
    (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
]


def create_landmarker(variant="lite"):
    """variant: 'lite' (fast, for live use), 'full' or 'heavy' (more accurate, for offline)."""
    path = f"pose_landmarker_{variant}.task"
    if not os.path.exists(path):
        print(f"Downloading {variant} pose model...")
        urllib.request.urlretrieve(MODEL_URL.format(v=variant), path)
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(options)


def detect(landmarker, bgr_frame, ts_ms):
    """Returns (image_landmarks, world_landmarks) or None. ts_ms must increase."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = landmarker.detect_for_video(image, ts_ms)
    if res.pose_landmarks and res.pose_world_landmarks:
        return res.pose_landmarks[0], res.pose_world_landmarks[0]
    return None


def _angle(a, b, c):
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def joint_angles(landmarks, world, min_vis=0.5):
    """8 joint angles in degrees (3D). NaN where a joint's landmarks aren't visible."""
    pts = np.array([[l.x, l.y, l.z] for l in world])
    vis = np.array([l.visibility for l in landmarks])
    out = np.full(len(JOINTS), np.nan)
    for k, (a, b, c) in enumerate(JOINTS.values()):
        if min(vis[a], vis[b], vis[c]) >= min_vis:
            out[k] = _angle(pts[a], pts[b], pts[c])
    return out


def mirror_vector(vec):
    """Swap left and right joints (for mirror-style following)."""
    return np.asarray(vec).reshape(-1, 2)[:, ::-1].ravel()


def draw_skeleton(frame, landmarks, flip_x=False, color=(0, 255, 0)):
    h, w = frame.shape[:2]
    px = [(int((1 - l.x if flip_x else l.x) * w), int(l.y * h)) for l in landmarks]
    for i, j in CONNECTIONS:
        cv2.line(frame, px[i], px[j], color, 2)
    for idx in {i for pair in CONNECTIONS for i in pair}:
        cv2.circle(frame, px[idx], 4, (0, 0, 255), -1)


def put_text(frame, text, y, color=(255, 255, 255), x=10, scale=0.6):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)
