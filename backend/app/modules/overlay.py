"""Shared live-overlay drawing helpers (ppe2-style boxes + labels)."""

from __future__ import annotations

from typing import Any, Iterable

import cv2
import numpy as np

LABEL_COLORS = {
    "nohairnet": (0, 0, 255),
    "nomask": (0, 0, 255),
    "crowd": (0, 0, 255),
    "loitering": (0, 140, 255),
    "frisking_missed": (0, 0, 255),
    "sling_psychrometer": (0, 220, 0),
    "no_sling_motion": (0, 165, 255),
}


def set_overlays(ctx_state: dict[str, Any], overlays: list[dict[str, Any]]) -> None:
    """Replace camera live overlays. Each item: label, box [x1,y1,x2,y2], score."""
    ctx_state["overlays"] = list(overlays or [])
    ctx_state["overlays_at"] = __import__("time").time()


def append_overlays(ctx_state: dict[str, Any], overlays: Iterable[dict[str, Any]]) -> None:
    bucket = ctx_state.setdefault("overlays", [])
    bucket.extend(list(overlays or []))
    ctx_state["overlays_at"] = __import__("time").time()


def draw_overlays(frame: np.ndarray, overlays: list[dict[str, Any]] | None) -> np.ndarray:
    """Draw detection boxes onto a copy of frame (ppe2 draw_label_detections style)."""
    if frame is None:
        return frame
    if not overlays:
        return frame
    out = frame.copy()
    for item in overlays:
        label = str(item.get("label") or "det")
        box = item.get("box") or item.get("bbox")
        score = float(item.get("score") or 0.0)
        if not box or len(box) < 4:
            continue
        color = tuple(item.get("color") or LABEL_COLORS.get(label, (0, 255, 0)))
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {score:.2f}" if score > 0 else label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
        text_w, text_h = text_size
        label_top = max(0, y1 - text_h - baseline - 6)
        cv2.rectangle(
            out,
            (x1, label_top),
            (x1 + text_w + 6, label_top + text_h + baseline + 6),
            color,
            -1,
        )
        cv2.putText(
            out,
            text,
            (x1 + 3, label_top + text_h + 3),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )
    return out
