"""HSV label / colored-object detection.

Kept separate from full_pipeline so tuning scripts do not import pyserial.
Uses fields from ``vision.parameters``.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from . import parameters


def _in_range_hsv(hsv, lower: tuple[int, int, int], upper: tuple[int, int, int]):
    """HSV range that supports hue wrap when lower_h > upper_h (common for red)."""
    lo = np.array(lower, dtype=np.uint8)
    hi = np.array(upper, dtype=np.uint8)
    if int(lo[0]) <= int(hi[0]):
        return cv2.inRange(hsv, lo, hi)
    m1 = cv2.inRange(hsv, lo, np.array((179, int(hi[1]), int(hi[2])), dtype=np.uint8))
    m2 = cv2.inRange(hsv, np.array((0, int(lo[1]), int(lo[2])), dtype=np.uint8), hi)
    return cv2.bitwise_or(m1, m2)


def _build_label_mask(hsv) -> np.ndarray:
    ranges = getattr(parameters, "hsv_ranges", None)
    if ranges:
        mask = None
        for low, high in ranges:
            m = _in_range_hsv(hsv, tuple(low), tuple(high))
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        return mask if mask is not None else np.zeros(hsv.shape[:2], dtype=np.uint8)

    lower = np.array(getattr(parameters, "hsv_lower", (20, 120, 120)), dtype=np.uint8)
    upper = np.array(getattr(parameters, "hsv_upper", (40, 255, 255)), dtype=np.uint8)
    if not getattr(parameters, "hsv_legacy_hue_wrap", False) and int(lower[0]) > int(
        upper[0]
    ):
        lower = np.array(lower, copy=True)
        upper = np.array(upper, copy=True)
        lower[0], upper[0] = upper[0], lower[0]
    return _in_range_hsv(hsv, tuple(lower.tolist()), tuple(upper.tolist()))


def _bbox_expand_intersects(
    ax: int,
    ay: int,
    aw: int,
    ah: int,
    bx: int,
    by: int,
    bw: int,
    bh: int,
    gap: int,
) -> bool:
    """Whether axis-aligned rects intersect after expanding both by gap (pixels)."""
    ax0, ay0 = ax - gap, ay - gap
    ax1, ay1 = ax + aw + gap, ay + ah + gap
    bx0, by0 = bx - gap, by - gap
    bx1, by1 = bx + bw + gap, by + bh + gap
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


def _merge_contours_union(contours, fh: int, fw: int):
    """Draw filled union of contours and return outer boundary contour + mask."""
    u = np.zeros((fh, fw), dtype=np.uint8)
    for c in contours:
        cv2.drawContours(u, [c], -1, 255, thickness=-1)
    outer, _ = cv2.findContours(u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    merged = outer[0] if outer else np.empty((0, 1, 2), dtype=np.int32)
    return merged, u


def detect_colored_ball(frame):
    """Return (found, info) using HSV mask + contour scoring.

    Despite the name, this works for bottles when ``use_ball_shape_filters`` is
    False in parameters.py.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = _build_label_mask(hsv)
    fh, fw = hsv.shape[:2]
    frame_area = float(fh * fw)

    ksz = max(3, int(getattr(parameters, "morph_kernel_px", 5)))
    if ksz % 2 == 0:
        ksz += 1
    kernel = np.ones((ksz, ksz), dtype=np.uint8)
    o_it = max(0, int(getattr(parameters, "morph_open_iterations", 1)))
    c_it = max(0, int(getattr(parameters, "morph_close_iterations", 2)))
    if o_it > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=o_it)
    if c_it > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=c_it)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, {"mask": mask}

    best = None
    best_area = -1.0

    min_area = float(getattr(parameters, "min_contour_area", 500))
    frag_min_area = float(
        getattr(parameters, "merge_fragment_min_contour_area", max(320.0, min_area * 0.15))
    )
    frag_min_area = max(80.0, min(frag_min_area, min_area - 1.0))
    merge_on = bool(getattr(parameters, "merge_nearby_label_contours", False))
    merge_gap = max(0, int(getattr(parameters, "merge_max_gap_px", 36)))

    use_shape = bool(getattr(parameters, "use_ball_shape_filters", True))
    min_circularity = float(getattr(parameters, "min_circularity", 0.60))
    max_aspect = float(getattr(parameters, "max_bbox_aspect_ratio", 1.35))
    max_aspect = max(1.01, max_aspect)
    min_solidity = float(getattr(parameters, "min_solidity", 0.85))

    reject_border = bool(getattr(parameters, "reject_border_touching_contours", False))
    border_m = int(getattr(parameters, "border_margin_px", 4))
    max_bbox_frac = float(getattr(parameters, "max_bbox_area_frac", 1.0))
    max_bbox_w_frac = float(getattr(parameters, "max_bbox_width_frac", 1.0))
    max_bbox_h_frac = float(getattr(parameters, "max_bbox_height_frac", 1.0))
    max_contour_frac = float(getattr(parameters, "max_contour_area_frac", 1.0))

    min_blob_s = float(getattr(parameters, "min_blob_mean_saturation", 0.0))

    reject_big_low_sat = bool(getattr(parameters, "reject_large_low_sat_blobs", False))
    big_bbox_frac = float(getattr(parameters, "large_blob_min_bbox_area_frac", 0.12))
    big_mean_s_max = float(getattr(parameters, "large_blob_max_mean_saturation", 45.0))

    def passes_common_filters(cx, cy, cw, ch, cmask_arg, contour_frac):
        _bbox_area = float(cw * ch)
        _bbox_frac = _bbox_area / frame_area if frame_area > 0 else 0.0
        if max_bbox_frac < 1.0 and _bbox_frac > max_bbox_frac:
            return False
        if max_bbox_w_frac < 1.0 and fw > 0 and (float(cw) / float(fw)) > max_bbox_w_frac:
            return False
        if max_bbox_h_frac < 1.0 and fh > 0 and (float(ch) / float(fh)) > max_bbox_h_frac:
            return False
        if max_contour_frac < 1.0 and contour_frac > max_contour_frac:
            return False
        if reject_border:
            if (
                cx <= border_m
                or cy <= border_m
                or cx + cw >= fw - border_m
                or cy + ch >= fh - border_m
            ):
                return False
        _mean_s = cv2.mean(hsv[:, :, 1], mask=cmask_arg)[0]
        if min_blob_s > 0.0 and _mean_s < min_blob_s:
            return False
        if reject_big_low_sat and _bbox_frac >= big_bbox_frac:
            if _mean_s <= big_mean_s_max:
                return False
        return True

    cand_rects = []
    cand_contours = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < frag_min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        contour_frac = area / frame_area if frame_area > 0 else 0.0
        cmask = np.zeros((fh, fw), dtype=np.uint8)
        cv2.drawContours(cmask, [c], -1, 255, thickness=-1)
        if not passes_common_filters(x, y, w, h, cmask, contour_frac):
            continue

        if area < min_area and not merge_on:
            continue
        # With merge enabled, fragments in [frag_min_area, min_area) can join siblings.

        if use_shape:
            perimeter = cv2.arcLength(c, True)
            if perimeter <= 1e-6:
                continue
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            if circularity < min_circularity:
                continue
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 1e-6:
                continue
            solidity = area / hull_area
            if solidity < min_solidity:
                continue
            aspect = float(w) / float(h) if h > 0 else 999.0
            if aspect > max_aspect or (1.0 / aspect) > max_aspect:
                continue

        cand_rects.append((x, y, w, h))
        cand_contours.append(c)

    if cand_contours:
        if merge_on:
            n = len(cand_contours)
            parent = list(range(n))

            def dsu_find(i):
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i

            def dsu_union(i, j):
                ri, rj = dsu_find(i), dsu_find(j)
                if ri != rj:
                    parent[rj] = ri

            for i in range(n):
                xi, yi, wi, hi = cand_rects[i]
                for j in range(i + 1, n):
                    xj, yj, wj, hj = cand_rects[j]
                    if _bbox_expand_intersects(xi, yi, wi, hi, xj, yj, wj, hj, merge_gap):
                        dsu_union(i, j)

            groups = {}
            for i in range(n):
                r = dsu_find(i)
                groups.setdefault(r, []).append(cand_contours[i])

            for gcs in groups.values():
                merged_c, merged_u = _merge_contours_union(gcs, fh, fw)
                if merged_c.size == 0:
                    continue
                mx, my, mw, mh = cv2.boundingRect(merged_c)
                merged_area = float(cv2.contourArea(merged_c))
                mcf = merged_area / frame_area if frame_area > 0 else 0.0
                merged_mask_bin = merged_u.astype(np.uint8)
                merged_mask_bin[merged_mask_bin > 0] = 255
                if merged_area < min_area:
                    continue
                if not passes_common_filters(mx, my, mw, mh, merged_mask_bin, mcf):
                    continue
                if merged_area > best_area:
                    best_area = merged_area
                    best = merged_c
        else:
            for c in cand_contours:
                a = cv2.contourArea(c)
                if a >= min_area and a > best_area:
                    best_area = a
                    best = c

    if best is None:
        return False, {"mask": mask}

    area = cv2.contourArea(best)
    perimeter = cv2.arcLength(best, True)
    circularity = (
        4.0 * math.pi * area / (perimeter * perimeter) if perimeter > 1e-6 else 0.0
    )
    hull = cv2.convexHull(best)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 1e-6 else 0.0
    x, y, w, h = cv2.boundingRect(best)
    center = (x + w // 2, y + h // 2)
    bbox_area = float(w * h)
    bbox_frac = bbox_area / frame_area if frame_area > 0 else 0.0
    bw_frac = float(w) / float(fw) if fw > 0 else 0.0
    bh_frac = float(h) / float(fh) if fh > 0 else 0.0
    contour_frac = area / frame_area if frame_area > 0 else 0.0
    blob_mask = np.zeros((fh, fw), dtype=np.uint8)
    cv2.drawContours(blob_mask, [best], -1, 255, thickness=-1)
    mean_s = cv2.mean(hsv[:, :, 1], mask=blob_mask)[0]
    return True, {
        "mask": mask,
        "area": area,
        "circularity": circularity,
        "solidity": solidity,
        "bbox": (x, y, w, h),
        "center": center,
        "mean_saturation": mean_s,
        "bbox_area_frac": bbox_frac,
        "bbox_w_frac": bw_frac,
        "bbox_h_frac": bh_frac,
        "contour_area_frac": contour_frac,
    }
