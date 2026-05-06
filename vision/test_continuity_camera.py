#!/usr/bin/env python3
"""Continuity Camera smoke test with index switching.

Usage (from repo root):
    python3 -m laptop_stack.test_continuity_camera
    python3 scripts/test_continuity_camera.py
    python3 -m laptop_stack.test_continuity_camera --start 1 --max-index 5

Behavior:
    - Opens a camera device (defaults: try indices 0..max until one works).
    - Live overlay shows current camera index + key hints.
    - Keys:
          0-9 : switch to that camera index (if opened successfully)
          n   : try next index in range 0..max_index
          p   : try previous index
          q   : quit

On macOS with Continuity Camera you often see two devices (built-in + iPhone).
Use 0 vs 1 to tell which feed is which, then set laptop_stack.parameters.camera_id in
laptop_stack/parameters.py for the Continuity feed.
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2


def try_open_camera(idx: int) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        if cap:
            cap.release()
        return None
    # Keep latency low on macOS webcam / Continuity backends when supported.
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    for _ in range(30):
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap
        time.sleep(0.02)
    cap.release()
    return None


def find_first_camera(max_index: int) -> tuple[cv2.VideoCapture | None, int | None]:
    for idx in range(max_index + 1):
        cap = try_open_camera(idx)
        if cap is not None:
            return cap, idx
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        metavar="IDX",
        help="Open this index first instead of scanning from 0",
    )
    parser.add_argument(
        "--max-index",
        type=int,
        default=5,
        help="Maximum camera index to try when scanning",
    )
    parser.add_argument(
        "--read-retries",
        type=int,
        default=30,
        help="Per-frame read retries before reconnecting (Continuity can glitch)",
    )
    parser.add_argument(
        "--reconnect-attempts",
        type=int,
        default=20,
        help="How many reopen attempts before giving up after repeated read loss",
    )
    args = parser.parse_args()
    max_idx = max(0, args.max_index)

    cap: cv2.VideoCapture | None = None
    idx: int | None = None

    if args.start is not None:
        cap = try_open_camera(args.start)
        idx = args.start if cap is not None else None
        if cap is None:
            print(f"Could not open --start index {args.start}, scanning 0..{max_idx}")

    if cap is None:
        cap, idx = find_first_camera(max_idx)

    if cap is None:
        print("No camera opened. Enable Continuity Camera and grant camera access to Terminal/Cursor.")
        sys.exit(1)

    window = "continuity_camera_test"
    print(
        f"Using camera index {idx}. Overlay shows index.\n"
        "Keys: 0-9=switch camera, n=next, p=prev, q=quit"
    )

    def read_frame() -> tuple[bool, object | None]:
        for _ in range(max(1, args.read_retries)):
            ok, fr = cap.read()
            if ok and fr is not None:
                return True, fr
            time.sleep(0.01)
        return False, None

    def reopen_current() -> bool:
        nonlocal cap
        if idx is None:
            return False
        for attempt in range(max(1, args.reconnect_attempts)):
            cap.release()
            new_cap = try_open_camera(idx)
            if new_cap is not None:
                cap = new_cap
                print(f"Reopened camera index {idx} (attempt {attempt + 1})")
                return True
            time.sleep(0.05)
        print(f"Failed to reopen camera index {idx} after reconnect attempts")
        return False

    try:
        while True:
            ok, frame = read_frame()
            if not ok:
                print(f"Transient read loss on index {idx}, reconnecting...")
                if not reopen_current():
                    break
                continue

            vis = frame.copy()
            txt = f"camera_id={idx}  |  0-9 switch  n/p next/prev  q quit"
            cv2.putText(
                vis,
                txt,
                (10, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window, vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("n"):
                new_idx = min(max_idx, (idx or 0) + 1)
                new_cap = try_open_camera(new_idx)
                if new_cap is not None:
                    cap.release()
                    cap, idx = new_cap, new_idx
                    print(f"Switched to index {idx}")
                else:
                    print(f"Index {new_idx} not available")
            elif key == ord("p"):
                new_idx = max(0, (idx or 0) - 1)
                new_cap = try_open_camera(new_idx)
                if new_cap is not None:
                    cap.release()
                    cap, idx = new_cap, new_idx
                    print(f"Switched to index {idx}")
                else:
                    print(f"Index {new_idx} not available")
            elif ord("0") <= key <= ord("9"):
                new_idx = key - ord("0")
                if new_idx > max_idx:
                    print(f"Index {new_idx} above --max-index {max_idx}")
                else:
                    new_cap = try_open_camera(new_idx)
                    if new_cap is not None:
                        cap.release()
                        cap, idx = new_cap, new_idx
                        print(f"Switched to index {idx}")
                    else:
                        print(f"Index {new_idx} not available")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
