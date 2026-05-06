"""OpenCV VideoCapture helpers (standalone; no pyserial)."""

from __future__ import annotations

from typing import Any, Optional

import cv2


def _open_first_working_webcam(parameters: Any) -> tuple[Optional[cv2.VideoCapture], Optional[int]]:
    primary = int(getattr(parameters, "camera_id", 0))
    fallbacks = list(getattr(parameters, "camera_fallback_indices", [0, 2]))
    order: list[int] = []
    seen: set[int] = set()
    for idx in [primary] + fallbacks:
        if idx not in seen:
            seen.add(idx)
            order.append(idx)

    for idx in order:
        cap = cv2.VideoCapture(idx)
        if cap is None or not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            continue
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap, idx
        try:
            cap.release()
        except Exception:
            pass

    return None, None


def open_cv_camera(parameters: Any) -> tuple[Optional[cv2.VideoCapture], Optional[int]]:
    """Open URL or webcam by index.

    On macOS the Continuity index can disappear when the phone disconnects,
    leaving no device at ``camera_id``. We try fallback indices until one opens
    and returns a frame.

    If ``camera_stream_mode`` is ``url`` and the stream fails (timeout, wrong IP),
    we optionally fall back to local webcam indices (see
    ``camera_url_fallback_to_webcam``).
    """

    mode = getattr(parameters, "camera_stream_mode", "continuity")
    if mode == "url":
        url = getattr(parameters, "camera_stream_url", "")
        if not url:
            return _open_first_working_webcam(parameters)

        cap = cv2.VideoCapture(url)
        if cap is None or not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            if getattr(parameters, "camera_url_fallback_to_webcam", True):
                return _open_first_working_webcam(parameters)
            return None, None
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        ok, frame = cap.read()
        if not ok or frame is None:
            try:
                cap.release()
            except Exception:
                pass
            if getattr(parameters, "camera_url_fallback_to_webcam", True):
                return _open_first_working_webcam(parameters)
            return None, None
        return cap, -1

    return _open_first_working_webcam(parameters)
