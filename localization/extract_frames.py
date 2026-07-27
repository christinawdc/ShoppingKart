"""Extract frames from a ceiling walkthrough video at a fixed time interval.

Writes frames to dataset/frames/ and an index to dataset/labels.csv.

    python localization/extract_frames.py assets/stock.mp4
    python localization/extract_frames.py assets/stock.mp4 --interval 0.25
"""

import argparse
import csv
from pathlib import Path

import cv2


def sharpness(frame):
    """Variance of the Laplacian. Higher = sharper, lower = more motion blur."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def blown_out(frame):
    """Fraction of pixels that are fully saturated -- the LED strip problem."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float((gray >= 250).mean())


def parse_skip(spec):
    """'7.6-8.9' -> (7.6, 8.9). Time ranges to drop from the source video."""
    start, _, end = spec.partition("-")
    return float(start), float(end)


def extract(video, out_dir, interval, max_width, rotate, skips):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"could not open {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps * interval))

    print(f"{video.name}: {total} frames @ {fps:.1f}fps "
          f"({total / fps:.1f}s) -> every {step} frames ({interval}s)")

    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    src_idx = 0
    kept = 0
    size = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t = src_idx / fps
        if src_idx % step == 0 and not any(a <= t <= b for a, b in skips):
            if rotate:
                frame = cv2.rotate(frame, {
                    90: cv2.ROTATE_90_CLOCKWISE,
                    180: cv2.ROTATE_180,
                    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
                }[rotate])

            if max_width and frame.shape[1] > max_width:
                scale = max_width / frame.shape[1]
                frame = cv2.resize(frame, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)

            size = frame.shape[:2]
            name = f"frame_{kept:04d}.jpg"
            cv2.imwrite(str(frames_dir / name), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

            rows.append({
                "frame": kept,
                "filename": name,
                "time_sec": round(t, 3),
                "sharpness": round(sharpness(frame), 1),
                "blown_out": round(blown_out(frame), 4),
                "label": "",
            })
            kept += 1

        src_idx += 1

    cap.release()

    if not rows:
        raise SystemExit("no frames extracted -- is the video readable?")

    csv_path = out_dir / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    h, w = size
    sharp = sorted(r["sharpness"] for r in rows)
    blown = sorted(r["blown_out"] for r in rows)

    print(f"\nwrote {kept} frames at {w}x{h} -> {frames_dir}")
    print(f"index -> {csv_path}")
    print(f"\nsharpness  min {sharp[0]:8.1f}   median {sharp[len(sharp) // 2]:8.1f}   "
          f"max {sharp[-1]:8.1f}")
    print(f"blown out  min {blown[0]:8.1%}   median {blown[len(blown) // 2]:8.1%}   "
          f"max {blown[-1]:8.1%}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video", type=Path)
    p.add_argument("--interval", type=float, default=0.5,
                   help="seconds between kept frames (default: 0.5)")
    p.add_argument("--out", type=Path, default=Path("dataset"))
    p.add_argument("--max-width", type=int, default=720,
                   help="downscale wider frames (default: 720, 0 to disable)")
    p.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0,
                   help="rotate frames if OpenCV ignores the video's rotation flag")
    p.add_argument("--skip", action="append", default=[], metavar="START-END",
                   help="drop a time range in seconds, e.g. --skip 7.6-8.9 (repeatable)")
    args = p.parse_args()

    skips = [parse_skip(s) for s in args.skip]
    if skips:
        print("skipping: " + ", ".join(f"{a}-{b}s" for a, b in skips))

    extract(args.video, args.out, args.interval, args.max_width, args.rotate, skips)


if __name__ == "__main__":
    main()
