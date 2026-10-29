"""
Extract reference poses from a Tai Chi video.

Usage:
    python extract_reference.py "https://www.youtube.com/watch?v=M4HsTHTUS-0"
    python extract_reference.py --video my_clip.mp4 --start 30 --end 150

Output (in ./reference/):
    reference.json   keyframe joint angles + the full angle sequence (for later DTW work)
    kf_00.jpg ...    thumbnails of each keyframe with the detected skeleton, for REVIEW
"""
import argparse
import json
import os

import cv2
import numpy as np

from pose_common import (CONNECTIONS, JOINT_NAMES, create_landmarker, detect,
                         joint_angles, mirror_vector)


def download(url, out_dir):
    import yt_dlp  # pip install yt-dlp
    opts = {
        # best single file (no ffmpeg merge needed), preferring <= 720p
        "format": "b[height<=720]/b",
        "outtmpl": os.path.join(out_dir, "source.%(ext)s"),
        "quiet": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info), info.get("title", "")
    except yt_dlp.utils.DownloadError:
        raise SystemExit(
            "\nYouTube download failed. Usually yt-dlp is outdated or has no JavaScript runtime.\n"
            "Fix: pip install -U yt-dlp  and  winget install DenoLand.Deno  (then reopen the terminal),\n"
            "or download the video yourself and run:\n"
            "    python extract_reference.py --video path\\to\\video.mp4"
        )


def analyze(path, landmarker, target_fps, start, end):
    """Run pose detection on ~target_fps frames. Returns times, frame indices, angles, landmarks."""
    if not os.path.exists(path):
        raise SystemExit(f"Video file not found: {path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"OpenCV could not open '{path}'. Re-encode it to H.264 (see README note / ffmpeg command).")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="replace")
    print(f"Video: {src_fps:.1f} fps, {n_frames} frames, codec tag '{fourcc}'")
    step = max(1, round(src_fps / target_fps))
    times, fidx, angles, lms = [], [], [], []
    i = 0
    while True:
        t = i / src_fps
        if t > end:
            break
        if t < start or i % step:
            if not cap.grab():
                break
            i += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        res = detect(landmarker, frame, int(t * 1000))
        times.append(t)
        fidx.append(i)
        if res:
            angles.append(joint_angles(*res))
            lms.append(np.array([[l.x, l.y] for l in res[0]]))
        else:
            angles.append(np.full(len(JOINT_NAMES), np.nan))
            lms.append(None)
        i += 1
        if len(times) % 50 == 0:
            print(f"  processed {len(times)} frames (t={t:.0f}s)")
    cap.release()
    if not times:
        raise SystemExit(
            "No frames could be decoded from the video (likely an AV1/VP9 stream OpenCV can't read,\n"
            "or --start is past the end). Re-encode to H.264:\n"
            f'    ffmpeg -i "{path}" -c:v libx264 -an -vf scale=-2:720 reference/source_h264.mp4\n'
            "then run: python extract_reference.py --video reference/source_h264.mp4"
        )
    return np.array(times), fidx, np.array(angles), lms, src_fps / step


def fill_nans(A):
    """Linearly interpolate gaps. Joints that are (almost) never visible stay NaN."""
    A = A.copy()
    x = np.arange(len(A))
    for k in range(A.shape[1]):
        good = ~np.isnan(A[:, k])
        if good.sum() >= 2:
            A[:, k] = np.interp(x, x[good], A[good, k])
    return A


def smooth(A, win=5):
    pad = win // 2
    out = np.empty_like(A)
    for k in range(A.shape[1]):
        out[:, k] = np.convolve(np.pad(A[:, k], pad, mode="edge"), np.ones(win) / win, mode="valid")
    return out


def pick_keyframes(A_s, detected, fps, min_gap, dedupe, percentile):
    """Keyframes = moments where the body is moving slowest (poses are 'held' or turning points)."""
    d = np.diff(A_s, axis=0)
    vel = np.sqrt(np.nansum(d ** 2, axis=1)) * fps  # deg/s across all joints
    vel = np.concatenate([[vel[0]], vel])
    thr = np.percentile(vel[detected], percentile)
    w = max(1, int(0.5 * fps))
    cands = []
    for i in range(len(vel)):
        if not detected[i] or vel[i] > thr:
            continue
        lo, hi = max(0, i - w), min(len(vel), i + w + 1)
        if vel[i] <= vel[lo:hi].min():
            cands.append(i)
    cands.sort(key=lambda i: vel[i])  # slowest first
    chosen = []
    for i in cands:
        if all(abs(i - j) >= min_gap * fps for j in chosen):
            chosen.append(i)
    chosen.sort()
    kept = []
    for i in chosen:  # drop keyframes nearly identical to the previous one
        if kept and np.nanmean(np.abs(A_s[i] - A_s[kept[-1]])) < dedupe:
            continue
        kept.append(i)
    return kept


def to_list(row):
    return [None if np.isnan(v) else round(float(v), 1) for v in row]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?", help="YouTube URL")
    ap.add_argument("--video", help="use a local video file instead of downloading")
    ap.add_argument("--out", default="reference")
    ap.add_argument("--fps", type=float, default=10, help="analysis frame rate")
    ap.add_argument("--model", default="full", choices=["lite", "full", "heavy"])
    ap.add_argument("--start", type=float, default=0, help="start time (s)")
    ap.add_argument("--end", type=float, default=1e9, help="end time (s)")
    ap.add_argument("--mirror-reference", action="store_true",
                    help="swap left/right so the student copies the instructor like a mirror "
                         "(instructor's right arm <-> student's left arm)")
    ap.add_argument("--min-gap", type=float, default=2.0, help="min seconds between keyframes")
    ap.add_argument("--dedupe", type=float, default=12.0,
                    help="skip keyframes within this mean angle diff (deg) of the previous one")
    ap.add_argument("--percentile", type=float, default=35,
                    help="only consider moments slower than this speed percentile")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    title = ""
    if args.video:
        path = args.video
    elif args.url:
        print("Downloading video...")
        path, title = download(args.url, args.out)
    else:
        ap.error("give a YouTube URL or --video")

    print("Detecting poses...")
    landmarker = create_landmarker(args.model)
    times, fidx, A, lms, fps = analyze(path, landmarker, args.fps, args.start, args.end)
    detected = np.array([l is not None for l in lms])
    print(f"Pose detected in {detected.sum()}/{len(detected)} frames")
    if detected.sum() < 10:
        raise SystemExit("Too few frames with a detected person. Is the instructor fully visible?")

    if args.mirror_reference:
        A = np.array([mirror_vector(r) for r in A])
    A_s = smooth(fill_nans(A))
    keys = pick_keyframes(A_s, detected, fps, args.min_gap, args.dedupe, args.percentile)
    print(f"Selected {len(keys)} keyframes")

    # Second pass: save annotated thumbnails of the keyframes for review.
    cap = cv2.VideoCapture(path)
    keyframes = []
    for n, i in enumerate(keys):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx[i])
        ok, frame = cap.read()
        thumb_name = f"kf_{n:02d}.jpg"
        if ok:
            h, w = frame.shape[:2]
            lm = lms[i]
            for a, b in CONNECTIONS:
                cv2.line(frame, (int(lm[a][0] * w), int(lm[a][1] * h)),
                         (int(lm[b][0] * w), int(lm[b][1] * h)), (0, 255, 0), 3)
            cv2.putText(frame, f"#{n}  t={times[i]:.1f}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.imwrite(os.path.join(args.out, thumb_name), frame)
        keyframes.append({"id": n, "time_s": round(float(times[i]), 2),
                          "angles": to_list(A_s[i]), "thumbnail": thumb_name})
        print(f"  keyframe {n}: t={times[i]:.1f}s")
    cap.release()

    ref = {
        "source": args.url or args.video,
        "title": title,
        "joint_names": JOINT_NAMES,
        "mirrored": args.mirror_reference,
        "fps": fps,
        "keyframes": keyframes,
        "sequence": {"times": [round(float(t), 2) for t in times],
                     "angles": [to_list(r) for r in A_s]},
    }
    with open(os.path.join(args.out, "reference.json"), "w") as f:
        json.dump(ref, f)
    print(f"Saved {args.out}/reference.json - open the kf_*.jpg thumbnails and check they look right.")


if __name__ == "__main__":
    main()