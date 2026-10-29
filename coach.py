"""
Live Tai Chi coach: steps through the reference keyframes and tells you how to fix each pose.

Usage:
    python coach.py                      # uses reference/reference.json
    python coach.py --threshold 25       # be more forgiving (mean angle error in degrees)

Keys (click the video window first):
    r restart   n skip to next pose   p previous pose   q quit
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from pose_common import (JOINT_NAMES, create_landmarker, detect, draw_skeleton,
                         joint_angles, put_text)


def cue(name, diff):
    """diff = live - reference. A bigger elbow/knee/hip angle means straighter."""
    side = "left" if name.startswith("L_") else "right"
    part = name[2:]
    if part == "shoulder":
        return f"{'lower' if diff > 0 else 'raise'} your {side} arm"
    return f"{'bend' if diff > 0 else 'straighten'} your {side} {part}"


def overlay_thumb(frame, thumb, width=240):
    h, w = frame.shape[:2]
    th = min(int(thumb.shape[0] * width / thumb.shape[1]), h)
    frame[0:th, w - width:w] = cv2.resize(thumb, (width, int(thumb.shape[0] * width / thumb.shape[1])))[:th]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="reference/reference.json")
    ap.add_argument("--threshold", type=float, default=20.0,
                    help="max mean joint-angle error (deg) to count as a match")
    ap.add_argument("--hold", type=float, default=0.6, help="seconds the pose must be held")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    with open(args.ref) as f:
        ref = json.load(f)
    assert ref["joint_names"] == JOINT_NAMES, "reference was made with different joints"
    ref_dir = os.path.dirname(args.ref)
    targets = [np.array([np.nan if v is None else v for v in k["angles"]]) for k in ref["keyframes"]]
    thumbs = []
    for k in ref["keyframes"]:
        img = cv2.imread(os.path.join(ref_dir, k["thumbnail"]))
        # The coach view is a mirror. In same-side mode, flip the thumbnail so it looks like
        # what you should see; in mirror mode it already matches.
        if img is not None and not ref.get("mirrored"):
            img = cv2.flip(img, 1)
        thumbs.append(img)
    if not targets:
        raise SystemExit("Reference has no keyframes.")

    landmarker = create_landmarker("lite")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    idx, hold_start, flash_until, t0 = 0, None, 0.0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = detect(landmarker, frame, int((time.time() - t0) * 1000))  # unflipped: true L/R
        view = cv2.flip(frame, 1)  # mirror view for the student
        now = time.time()
        put_text(view, f"Pose {min(idx + 1, len(targets))}/{len(targets)}", 25)

        if idx >= len(targets):
            put_text(view, "Sequence complete!  (press r to restart)", 55, (0, 255, 0), scale=0.8)
        else:
            if thumbs[idx] is not None:
                overlay_thumb(view, thumbs[idx])
            if res is None:
                put_text(view, "No person detected - step back so your whole body is visible", 55, (0, 0, 255))
                hold_start = None
            else:
                landmarks, world = res
                live = joint_angles(landmarks, world)
                err = np.abs(live - targets[idx])
                valid = ~np.isnan(err)
                if valid.sum() < 4:
                    put_text(view, "Can't see enough joints - step back / improve lighting", 55, (0, 0, 255))
                    hold_start = None
                    draw_skeleton(view, landmarks, flip_x=True, color=(0, 165, 255))
                else:
                    mean_err, max_err = err[valid].mean(), err[valid].max()
                    matched = mean_err < args.threshold and max_err < 2 * args.threshold
                    color = (0, 255, 0) if matched else (0, 165, 255)
                    draw_skeleton(view, landmarks, flip_x=True, color=color)
                    put_text(view, f"Error: {mean_err:.0f} deg (need < {args.threshold:.0f})", 55, color)

                    if matched:
                        hold_start = hold_start or now
                        put_text(view, "Good - hold it...", 85, (0, 255, 0))
                        cv2.rectangle(view, (10, 95), (10 + int(200 * min(1, (now - hold_start) / args.hold)), 105),
                                      (0, 255, 0), -1)
                        if now - hold_start >= args.hold:
                            idx += 1
                            hold_start = None
                            flash_until = now + 1.0
                    else:
                        hold_start = None
                        worst = np.argsort(np.where(valid, err, -1))[::-1][:2]
                        y = 85
                        for j in worst:
                            if valid[j] and err[j] > args.threshold * 0.75:
                                put_text(view, f"- {cue(JOINT_NAMES[j], live[j] - targets[idx][j])}", y, (0, 200, 255))
                                y += 25

        if now < flash_until:
            put_text(view, "Pose matched!", view.shape[0] - 20, (0, 255, 0), scale=1.0)

        cv2.imshow("Tai Chi coach", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            idx, hold_start = 0, None
        elif key == ord("n"):
            idx, hold_start = min(idx + 1, len(targets)), None
        elif key == ord("p"):
            idx, hold_start = max(idx - 1, 0), None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
