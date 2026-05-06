#!/usr/bin/env python3
"""Tune HSV + shape filters using the live Continuity Camera feed.

Uses trackbars live. Run from repo root:
  python3 -m vision.tune_ball_hsv
  python3 -m vision.tune_ball_hsv --url 'http://PHONE_IP:4747/video'
  python3 -m vision.tune_ball_hsv --list-cameras
  python3 -m vision.tune_ball_hsv --continuity --camera-id 1

Press:
  s = print current slider values → paste into vision/parameters.py
  q = quit

Note: this tool edits the legacy single-box fields ``hsv_lower`` and
``hsv_upper``. If ``parameters.hsv_ranges`` is non-empty, detection ORs those
ranges together and this preview may not match unless you temporarily set
``hsv_ranges = []`` while tuning one band at a time.
"""

from __future__ import annotations

import argparse
import sys
import time
import numpy as np

import cv2

from .camera_util import open_cv_camera
from . import parameters
from .vision_detection import detect_colored_ball


def _probe_camera_indices(max_idx: int = 8) -> None:
    print(f"OpenCV probing indices 0..{max_idx - 1} (first usable may be Continuity / Mac cam):", flush=True)
    for i in range(max_idx):
        cap = cv2.VideoCapture(i)
        if cap is None or not cap.isOpened():
            print(f"  [{i}] not available", flush=True)
            try:
                cap.release()
            except Exception:
                pass
            continue
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        ok, fr = cap.read()
        try:
            cap.release()
        except Exception:
            pass
        if ok and fr is not None:
            print(f"  [{i}] OK frame shape {fr.shape}", flush=True)
        else:
            print(f"  [{i}] opened but read failed", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Live HSV + shape tuner for colored blob detection.")
    ap.add_argument(
        "--url",
        metavar="URL",
        default=None,
        help="Phone stream URL (overrides parameters.camera_stream_url), e.g. http://192.168.1.42:8080/video",
    )
    ap.add_argument(
        "--continuity",
        action="store_true",
        help="Use webcam / Continuity indices from parameters.camera_id (+ fallbacks)",
    )
    ap.add_argument("--camera-id", type=int, default=None, metavar="N", help="With --continuity, try this index first")
    ap.add_argument(
        "--allow-webcam-fallback",
        action="store_true",
        help="If --url fails, fall back to laptop webcam (otherwise exit with error)",
    )
    ap.add_argument("--list-cameras", action="store_true", help="Print which camera indices yield frames, then exit")
    ns = ap.parse_args()

    if ns.list_cameras:
        _probe_camera_indices()
        sys.exit(0)

    if ns.url:
        parameters.camera_stream_url = ns.url
        parameters.camera_stream_mode = "url"
    if ns.continuity:
        parameters.camera_stream_mode = "continuity"
    if ns.camera_id is not None:
        parameters.camera_id = ns.camera_id
    if ns.allow_webcam_fallback:
        parameters.camera_url_fallback_to_webcam = True
    elif getattr(parameters, "camera_stream_mode", "") == "url":
        parameters.camera_url_fallback_to_webcam = False

    print(
        f"Camera: mode={getattr(parameters, 'camera_stream_mode', '?')} "
        f"url={getattr(parameters, 'camera_stream_url', '')!r} "
        f"allow_webcam_fallback={getattr(parameters, 'camera_url_fallback_to_webcam', False)}",
        flush=True,
    )

    # Trackbars MUST live on the same HighGUI window you imshow, or macOS opens a
    # second minimal window that is easy to miss behind the preview.
    win_name = "tune_ball_hsv (sliders below image)"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    print(
        "One OpenCV window should show BOTH the split preview and sliders under it.\n"
        "Scroll the window vertically if sliders are clipped.\n"
        "Left pane = camera, right pane = binary mask (white = in-range).\n\n"
        "Key s prints a paste-ready block to THIS terminal (nothing on screen).\n"
        "Click the OpenCV window first so q / s are received.",
        flush=True,
    )

    tb = lambda n, pos, mx, cb=lambda _: None: cv2.createTrackbar(n, win_name, pos, mx, cb)
    tb("h_min", parameters.hsv_lower[0], 179)
    tb("s_min", parameters.hsv_lower[1], 255)
    tb("v_min", parameters.hsv_lower[2], 255)
    tb("h_max", parameters.hsv_upper[0], 179)
    tb("s_max", parameters.hsv_upper[1], 255)
    tb("v_max", parameters.hsv_upper[2], 255)
    tb("min_area", parameters.min_contour_area, 20000)
    tb("min_circ_pct", int(parameters.min_circularity * 100), 99)
    tb("max_aspect_pct", int(parameters.max_bbox_aspect_ratio * 100), 300)
    tb("min_solid_pct", int(parameters.min_solidity * 100), 100)
    tb("morph_kernel", parameters.morph_kernel_px, 31)
    tb("open_it", parameters.morph_open_iterations, 5)
    tb("close_it", parameters.morph_close_iterations, 10)

    cap, used_idx = open_cv_camera(parameters)
    if cap is None:
        print(
            "No camera opened.\n"
            "  • url mode: set camera_stream_url to the URL shown by the phone app; same Wi‑Fi as laptop;\n"
            "    try: curl -I '<your url>'\n"
            "  • continuity: tweak camera_id / camera_fallback_indices, reconnect phone as camera.",
            flush=True,
        )
        sys.exit(1)
    if used_idx >= 0:
        print(f"Using camera index {used_idx} (see parameters.camera_id / camera_fallback_indices).", flush=True)
        if getattr(parameters, "camera_stream_mode", "") == "url":
            print(
                "[WARN] URL camera failed or timed out; fell back to local webcam "
                "(fix camera_stream_url / Wi‑Fi / phone app or set "
                "camera_url_fallback_to_webcam = False to see errors only).",
                flush=True,
            )
    elif used_idx == -1:
        print("Using camera_stream_url (url mode).", flush=True)

    def read_frame_robust() -> tuple[bool, object | None]:
        for _ in range(40):
            ok, frame = cap.read()
            if ok and frame is not None:
                return True, frame
            time.sleep(0.01)
        return False, None

    ok0, preview = read_frame_robust()
    if not ok0 or preview is None:
        print("Camera opened but reads failed.")
        cap.release()
        sys.exit(1)

    saved_flash = 0

    while True:
        ok, frame = read_frame_robust()
        if not ok:
            frame = preview
        preview = frame

        h_min = cv2.getTrackbarPos("h_min", win_name)
        s_min = cv2.getTrackbarPos("s_min", win_name)
        v_min = cv2.getTrackbarPos("v_min", win_name)
        h_max = cv2.getTrackbarPos("h_max", win_name)
        s_max = cv2.getTrackbarPos("s_max", win_name)
        v_max = cv2.getTrackbarPos("v_max", win_name)

        if not getattr(parameters, "hsv_legacy_hue_wrap", False) and h_min > h_max:
            h_min, h_max = h_max, h_min
            cv2.setTrackbarPos("h_min", win_name, h_min)
            cv2.setTrackbarPos("h_max", win_name, h_max)

        parameters.hsv_lower = (h_min, s_min, v_min)
        parameters.hsv_upper = (h_max, s_max, v_max)
        parameters.min_contour_area = int(cv2.getTrackbarPos("min_area", win_name))
        parameters.min_circularity = max(
            0.01,
            float(cv2.getTrackbarPos("min_circ_pct", win_name)) / 100.0,
        )
        parameters.max_bbox_aspect_ratio = max(
            1.01,
            float(cv2.getTrackbarPos("max_aspect_pct", win_name)) / 100.0,
        )
        parameters.min_solidity = min(
            1.0,
            max(
                0.0,
                float(cv2.getTrackbarPos("min_solid_pct", win_name)) / 100.0,
            ),
        )

        ksz = cv2.getTrackbarPos("morph_kernel", win_name)
        if ksz % 2 == 0:
            ksz += 1
        parameters.morph_kernel_px = max(3, min(ksz, 31))
        parameters.morph_open_iterations = cv2.getTrackbarPos("open_it", win_name)
        parameters.morph_close_iterations = cv2.getTrackbarPos("close_it", win_name)

        found, info = detect_colored_ball(frame)
        dbg = frame.copy()
        cv2.rectangle(dbg, (8, 8), (760, 70), (0, 0, 0), -1)
        cv2.putText(
            dbg,
            f"h:[{parameters.hsv_lower[0]},{parameters.hsv_upper[0]}]"
            f" s:[{parameters.hsv_lower[1]},{parameters.hsv_upper[1]}]"
            f" v:[{parameters.hsv_lower[2]},{parameters.hsv_upper[2]}]"
            f" | found={'yes' if found else 'no'}",
            (12, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        if found and "bbox" in info:
            x, y, w, h = info["bbox"]
            cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 3)
            mean_s = info.get("mean_saturation")
            ba = info.get("bbox_area_frac")
            cw = info.get("bbox_w_frac")
            ch = info.get("bbox_h_frac")
            ca = info.get("contour_area_frac")
            frac = ""
            if (
                mean_s is not None
                and ba is not None
                and cw is not None
                and ch is not None
                and ca is not None
            ):
                frac = f" ms={mean_s:.0f} ba={ba:.2f} w={cw:.2f} h={ch:.2f} ca={ca:.2f}"
            label = (
                f"area={info['area']:.0f} circ={info['circularity']:.2f}"
                f" solid={info.get('solidity', 0):.2f}"
                + frac
            )
            cv2.putText(
                dbg,
                label,
                (x, max(24, y - 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        mask_vis = None
        if "mask" in info:
            mask_vis = cv2.cvtColor(info["mask"], cv2.COLOR_GRAY2BGR)
            cv2.putText(
                mask_vis,
                "threshold mask (binary)",
                (10, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if mask_vis is None:
            show_bgr = dbg
        else:
            if mask_vis.shape[0] != dbg.shape[0]:
                scale = dbg.shape[0] / float(mask_vis.shape[0])
                mask_vis = cv2.resize(
                    mask_vis,
                    (int(mask_vis.shape[1] * scale), dbg.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            show_bgr = np.hstack([dbg, mask_vis])
        bar_h = 56
        pad = np.zeros((bar_h, show_bgr.shape[1], 3), dtype=np.uint8)
        pad[:] = (40, 40, 40)
        msg_bar = (
            "q quit  |  s = print snippet to TERMINAL  |  scroll down for sliders"
        )
        cv2.putText(
            pad,
            msg_bar,
            (12, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if saved_flash > 0:
            cv2.putText(
                pad,
                "Printed values to terminal (below).",
                (12, bar_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            saved_flash -= 1
        show_bgr = np.vstack([show_bgr, pad])
        cv2.imshow(win_name, show_bgr)

        key = chr(cv2.waitKey(1) & 0xFF)
        if key == "q":
            break
        if key == "s":
            block = (
                "Copy into vision/parameters.py:\n"
                f"hsv_lower = {tuple(parameters.hsv_lower)}\n"
                f"hsv_upper = {tuple(parameters.hsv_upper)}\n"
                f"min_contour_area = {parameters.min_contour_area}\n"
                f"min_circularity = {parameters.min_circularity:.2f}\n"
                f"max_bbox_aspect_ratio = {parameters.max_bbox_aspect_ratio:.2f}\n"
                f"min_solidity = {parameters.min_solidity:.2f}\n"
                f"morph_kernel_px = {parameters.morph_kernel_px}\n"
                f"morph_open_iterations = {parameters.morph_open_iterations}\n"
                f"morph_close_iterations = {parameters.morph_close_iterations}\n"
            )
            print(block, flush=True)
            saved_flash = 90

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
