"""Loitering — continuous presence (ported from crowd_loitering_bypass/app.py)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .base import CameraContext, DetectionEvent


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center(box):
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def _center_distance(a, b) -> float:
    ax, ay = _center(a)
    bx, by = _center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _smooth_bbox(old_bbox, new_bbox, alpha=0.65):
    if old_bbox is None:
        return list(new_bbox)
    return [alpha * float(new_bbox[i]) + (1.0 - alpha) * float(old_bbox[i]) for i in range(4)]


def _match_score(bbox_a, bbox_b, relink_iou=0.15, relink_center=280.0) -> float:
    iou = _iou(bbox_a, bbox_b)
    dist = _center_distance(bbox_a, bbox_b)
    if iou < relink_iou and dist > relink_center:
        return -1.0
    dist_score = max(0.0, 1.0 - dist / max(relink_center, 1.0))
    return (iou * 2.0) + dist_score


def _nms(detections, iou_thresh=0.45, near_center=120.0, near_iou=0.15):
    ordered = sorted(detections, key=lambda d: d[2], reverse=True)
    kept = []
    for yolo_id, bbox, score in ordered:
        drop = False
        for _, kb, _ in kept:
            if _iou(bbox, kb) >= iou_thresh:
                drop = True
                break
            if _center_distance(bbox, kb) <= near_center and _iou(bbox, kb) >= near_iou:
                drop = True
                break
        if not drop:
            kept.append((yolo_id, bbox, score))
    return kept


class LoiteringModule:
    id = "loitering"
    labels = ["loitering"]

    def __init__(self) -> None:
        self.model = None
        self.device = "cpu"
        self.config: dict[str, Any] = {}

    def load(self, model_path: str, device: str, config: dict[str, Any]) -> None:
        from ultralytics import YOLO

        self.device = device
        self.config = config or {}
        # Each camera gets its own YOLO instance so BoTSORT persist state does not mix.
        self._model_path = model_path
        self.model = YOLO(model_path)
        try:
            self.model.to(device)
        except Exception:
            self.device = "cpu"

    def unload(self) -> None:
        self.model = None

    def _state(self, ctx: CameraContext) -> dict[str, Any]:
        return ctx.state.setdefault(
            "loiter",
            {
                "person_slots": {},
                "yolo_to_person": {},
                "next_person_index": 1,
                "tentative_people": [],
                "model": None,
            },
        )

    def _camera_model(self, state: dict[str, Any]):
        """Per-camera tracker model (BoTSORT state isolation)."""
        from ultralytics import YOLO

        if state.get("model") is None:
            state["model"] = YOLO(self._model_path)
            try:
                state["model"].to(self.device)
            except Exception:
                pass
        return state["model"]

    def _presence_seconds(self, slot, now, miss_gap) -> float:
        started = slot.get("present_started_at")
        if started is None:
            return 0.0
        last_seen = float(slot.get("last_seen_at") or started)
        if now - last_seen <= miss_gap:
            return max(0.0, now - float(started))
        return max(0.0, last_seen - float(started))

    def _touch(self, state, person_key, bbox, yolo_id, now, miss_gap, smooth):
        slot = state["person_slots"][person_key]
        gap = now - float(slot["last_seen_at"])
        if gap > miss_gap:
            slot["present_started_at"] = now
        slot["last_seen_at"] = now
        slot["bbox"] = _smooth_bbox(slot.get("bbox"), bbox, smooth)
        slot["active"] = True
        if yolo_id is not None:
            yolo_id = int(yolo_id)
            old = slot.get("yolo_id")
            if old is not None and int(old) != yolo_id:
                state["yolo_to_person"].pop(int(old), None)
            slot["yolo_id"] = yolo_id
            state["yolo_to_person"][yolo_id] = person_key
        return slot

    def _confirm_new(self, state, pending, now, cfg):
        min_conf = float(cfg.get("new_person_min_confidence", 0.45))
        need = int(cfg.get("new_person_confirm_frames", 5))
        relink_iou = float(cfg.get("track_relink_iou", 0.15))
        relink_center = float(cfg.get("track_relink_center_px", 280))
        smooth = float(cfg.get("track_bbox_smooth", 0.65))
        miss_gap = float(cfg.get("miss_gap_seconds", 180))
        promoted = []
        tentatives = state["tentative_people"]

        for yolo_id, bbox, score in list(pending):
            if score < min_conf:
                continue
            best_idx, best_score = None, -1.0
            for i, tent in enumerate(tentatives):
                spatial = _match_score(tent["bbox"], bbox, relink_iou, relink_center)
                if spatial > best_score:
                    best_score, best_idx = spatial, i
            if best_idx is not None and best_score >= 0:
                tent = tentatives[best_idx]
                tent["hits"] += 1
                tent["bbox"] = _smooth_bbox(tent["bbox"], bbox, smooth)
                tent["last_seen"] = now
                tent["yolo_id"] = yolo_id
                tent["score"] = score
                if tent["hits"] >= need:
                    key = f"person_{state['next_person_index']}"
                    state["next_person_index"] += 1
                    state["person_slots"][key] = {
                        "present_started_at": tent["first_seen"],
                        "last_seen_at": now,
                        "uploaded_at": 0.0,
                        "bbox": tent["bbox"],
                        "yolo_id": yolo_id,
                        "active": True,
                    }
                    if yolo_id is not None:
                        state["yolo_to_person"][int(yolo_id)] = key
                    tentatives.pop(best_idx)
                    promoted.append((key, tent["bbox"], score, yolo_id))
            else:
                tentatives.append(
                    {
                        "bbox": list(bbox),
                        "hits": 1,
                        "first_seen": now,
                        "last_seen": now,
                        "yolo_id": yolo_id,
                        "score": score,
                    }
                )

        state["tentative_people"] = [
            t for t in tentatives if now - float(t["last_seen"]) <= 2.5 and t["hits"] > 0
        ]
        return promoted

    def _merge_slots(self, state, assignments, now, cfg):
        merge_iou = float(cfg.get("slot_merge_iou", 0.40))
        merge_center = float(cfg.get("slot_merge_center_px", 120))
        changed = True
        while changed:
            changed = False
            active_keys = [a[0] for a in assignments]
            for i in range(len(active_keys)):
                for j in range(i + 1, len(active_keys)):
                    a, b = active_keys[i], active_keys[j]
                    if a not in state["person_slots"] or b not in state["person_slots"]:
                        continue
                    sa, sb = state["person_slots"][a], state["person_slots"][b]
                    if _iou(sa["bbox"], sb["bbox"]) >= merge_iou or _center_distance(sa["bbox"], sb["bbox"]) <= merge_center:
                        keep, drop = (a, b) if int(a.split("_")[1]) <= int(b.split("_")[1]) else (b, a)
                        sk, sd = state["person_slots"][keep], state["person_slots"][drop]
                        sk["present_started_at"] = min(float(sk["present_started_at"]), float(sd["present_started_at"]))
                        if sd.get("uploaded_at") and (not sk.get("uploaded_at") or sd["uploaded_at"] < sk["uploaded_at"]):
                            sk["uploaded_at"] = sd["uploaded_at"]
                        for yid,pkey in list(state["yolo_to_person"].items()):
                            if pkey == drop:
                                state["yolo_to_person"][yid] = keep
                        state["person_slots"].pop(drop, None)
                        assignments = [
                            (keep if key == drop else key, bbox, score, yolo_id)
                            for key, bbox, score, yolo_id in assignments
                        ]
                        # dedupe same keep
                        seen = set()
                        uniq = []
                        for item in assignments:
                            if item[0] in seen:
                                continue
                            seen.add(item[0])
                            uniq.append(item)
                        assignments = uniq
                        changed = True
                        break
                if changed:
                    break
        return assignments

    def _associate(self, state, detections, now, cfg):
        miss_gap = float(cfg.get("miss_gap_seconds", 180))
        grace = float(cfg.get("track_lost_grace_seconds", 90))
        relink_iou = float(cfg.get("track_relink_iou", 0.15))
        relink_center = float(cfg.get("track_relink_center_px", 280))
        smooth = float(cfg.get("track_bbox_smooth", 0.65))
        assignments = []
        used = set()
        pending = []

        for yolo_id, bbox, score in detections:
            person_key = state["yolo_to_person"].get(int(yolo_id)) if yolo_id is not None else None
            if person_key and person_key in state["person_slots"] and person_key not in used:
                slot = state["person_slots"][person_key]
                if _match_score(slot["bbox"], bbox, relink_iou, relink_center) >= 0:
                    self._touch(state, person_key, bbox, yolo_id, now, miss_gap, smooth)
                    used.add(person_key)
                    assignments.append((person_key, bbox, score, yolo_id))
                    continue
            pending.append((yolo_id, bbox, score))

        candidates = [
            key
            for key, slot in state["person_slots"].items()
            if key not in used and now - float(slot["last_seen_at"]) <= grace
        ]
        while pending and candidates:
            best = None
            for det_index, (yolo_id, bbox, score) in enumerate(pending):
                for person_key in candidates:
                    spatial = _match_score(state["person_slots"][person_key]["bbox"], bbox, relink_iou, relink_center)
                    if spatial < 0:
                        continue
                    if best is None or spatial > best[0]:
                        best = (spatial, det_index, person_key)
            if best is None:
                break
            _, det_index, person_key = best
            yolo_id, bbox, score = pending.pop(det_index)
            candidates.remove(person_key)
            used.add(person_key)
            self._touch(state, person_key, bbox, yolo_id, now, miss_gap, smooth)
            assignments.append((person_key, bbox, score, yolo_id))

        promoted = self._confirm_new(state, pending, now, cfg)
        assignments.extend(promoted)
        assignments = self._merge_slots(state, assignments, now, cfg)
        used = {item[0] for item in assignments}
        for person_key, slot in state["person_slots"].items():
            slot["active"] = person_key in used
        return assignments

    def process(self, frame: np.ndarray, ctx: CameraContext) -> list[DetectionEvent]:
        if self.model is None:
            return []
        cfg = self.config
        threshold = float(cfg.get("threshold_seconds", 1800))
        conf = float(cfg.get("confidence", 0.35))
        cooldown = float(cfg.get("cooldown_seconds", 1800))
        grace = float(cfg.get("track_lost_grace_seconds", 90))
        tracker = cfg.get("tracker", "botsort.yaml")
        state = self._state(ctx)
        model = self._camera_model(state)
        now = time.time()

        results = model.track(
            frame,
            conf=conf,
            classes=[0],
            persist=True,
            tracker=tracker,
            verbose=False,
            device=self.device,
        )
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else [1.0] * len(boxes)
            ids = (
                result.boxes.id.int().cpu().tolist()
                if getattr(result.boxes, "id", None) is not None
                else [None] * len(boxes)
            )
            for box, tid, score in zip(boxes, ids, scores):
                detections.append((tid, [float(v) for v in box], float(score)))
        detections = _nms(detections)

        assignments = self._associate(state, detections, now, cfg)

        # prune lost slots
        for key in list(state["person_slots"].keys()):
            if now - float(state["person_slots"][key]["last_seen_at"]) > grace:
                yid = state["person_slots"][key].get("yolo_id")
                if yid is not None:
                    state["yolo_to_person"].pop(int(yid), None)
                state["person_slots"].pop(key, None)

        miss_gap = float(cfg.get("miss_gap_seconds", 180))
        events: list[DetectionEvent] = []
        for person_key, bbox, score, _yolo_id in assignments:
            slot = state["person_slots"].get(person_key)
            if not slot:
                continue
            present = self._presence_seconds(slot, now, miss_gap)
            can_upload = (not slot.get("uploaded_at")) or (now - float(slot["uploaded_at"]) >= cooldown)
            if present >= threshold and can_upload:
                slot["uploaded_at"] = now
                events.append(
                    DetectionEvent(
                        label="loitering",
                        module_id=self.id,
                        boxes=[bbox],
                        scores=[score],
                        detail={"present_seconds": round(present, 1), "track": person_key},
                        frame=frame.copy(),
                    )
                )
        ctx.state["active_tracks"] = len(assignments)
        return events
