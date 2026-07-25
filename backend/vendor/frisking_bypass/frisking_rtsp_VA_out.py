import cv2
import numpy as np
import os
import math
import base64
import json
import time
import requests
from collections import defaultdict, deque
from datetime import datetime
from os import getenv
from ultralytics import YOLO
from motion_tracking import KalmanBBoxTracker

# Import the security model strictly from the yolov8 module (not from app.py)
try:
    from yolov8_model.yolov8_api_demo import yolov8_detect_security
except ImportError:
    print("Warning: Could not import yolov8_detect_security. Security checks will be bypassed.")
    def yolov8_detect_security(frame, conf):
        return [], [], []

PREVIOUS_FRISKING_TRACK_EVENT_LABEL = 'frisking_missed'
SECURITY_PERSONNEL_TOL = float(getenv('SECURITY_PERSONNEL_TOL', '0.4'))
TRACKING_EVENT_FRAMES = int(getenv('TRACKING_EVENT_FRAMES', '10'))
TRACKING_EVENT_PRE_CROSSING_FRAMES = max(
    1, int(getenv('TRACKING_EVENT_PRE_CROSSING_FRAMES', '6'))
)
TRACKING_EVENT_POST_CROSSING_FRAMES = max(
    0, int(getenv('TRACKING_EVENT_POST_CROSSING_FRAMES', '3'))
)
TRACKING_EVENT_REQUIRE_COMPLETE_WINDOW = getenv(
    'TRACKING_EVENT_REQUIRE_COMPLETE_WINDOW', '1'
).lower() in ('1', 'true', 'yes', 'on')
TRACKING_EVENT_REQUIRE_SAME_TRACKER_ID = getenv(
    'TRACKING_EVENT_REQUIRE_SAME_TRACKER_ID', '1'
).lower() in ('1', 'true', 'yes', 'on')
TRACKING_HISTORY_SECONDS = float(getenv('TRACKING_HISTORY_SECONDS', '10.0'))
TRACKING_HISTORY_MAX_FRAMES = int(getenv('TRACKING_HISTORY_MAX_FRAMES', '180'))
TRACKING_FRAME_JPEG_QUALITY = int(getenv('TRACKING_FRAME_JPEG_QUALITY', '75'))
TRACKING_FRAME_MAX_WIDTH = int(getenv('TRACKING_FRAME_MAX_WIDTH', '960'))

# ─── API Upload Helpers ───────────────────────────────────────────────

def load_upload_config(config_path="upload_api_parameters.json"):
    """Load API upload configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded upload config: {config_path}")
        print(f"  Server: {config.get('server_url')}")
        print(f"  Camera: {config.get('camera_id')}")
        print(f"  Enabled: {config.get('enabled', False)}")
        return config
    except FileNotFoundError:
        print(f"Warning: {config_path} not found. API upload disabled.")
        return {"enabled": False}
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in {config_path}: {e}. API upload disabled.")
        return {"enabled": False}


def compress_image_to_bytes(frame, max_kb=900):
    """Compress a CV2 frame to JPEG bytes under max_kb size."""
    quality = 92
    while quality >= 10:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        success, buffer = cv2.imencode('.jpg', frame, encode_params)
        if not success:
            return None
        if len(buffer) / 1024 <= max_kb:
            return buffer.tobytes()
        quality -= 8

    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 10])
    return buffer.tobytes() if success else None


def generate_thumbnail(frame, size=(175, 175), max_kb=25):
    """Generate a small JPEG thumbnail from a CV2 frame."""
    h, w = frame.shape[:2]
    side = min(h, w)
    y_start = (h - side) // 2
    x_start = (w - side) // 2
    cropped = frame[y_start:y_start + side, x_start:x_start + side]
    resized = cv2.resize(cropped, size, interpolation=cv2.INTER_AREA)

    quality = 70
    while quality >= 5:
        success, buffer = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not success:
            return None
        if len(buffer) / 1024 <= max_kb:
            return buffer.tobytes()
        quality -= 10

    success, buffer = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 5])
    return buffer.tobytes() if success else None


def build_event_payload(camera_id, label, bbox_xyxy):
    """Build the event JSON matching the schema expected by route.js."""
    ts = time.time()
    event_id = f"{label}-{camera_id}-{int(ts * 1000)}"
    primary_bbox = list(bbox_xyxy)
    w = max(primary_bbox[2] - primary_bbox[0], 1)
    h = max(primary_bbox[3] - primary_bbox[1], 1)

    state = {
        "id": event_id,
        "camera": camera_id,
        "frame_time": ts,
        "snapshot": {
            "frame_time": ts,
            "box": primary_bbox,
            "area": w * h,
            "region": [0, 0, 1280, 720],
            "score": 1,
            "attributes": []
        },
        "label": label,
        "sub_label": None,
        "top_score": 1,
        "false_positive": False,
        "start_time": ts,
        "end_time": ts + 10,
        "score": 1,
        "box": primary_bbox,
        "boxes": [primary_bbox],
        "area": w * h,
        "ratio": round(w / h, 2),
        "region": [0, 0, 1280, 720],
        "stationary": False,
        "motionless_count": 0,
        "position_changes": 1,
        "current_zones": [],
        "entered_zones": [],
        "has_clip": False,
        "has_snapshot": True,
        "attributes": {},
        "current_attributes": []
    }

    return {
        "before": {**state},
        "after": {**state},
        "type": "end"
    }


def upload_event_to_server(upload_config, event_frame, bbox_xyxy, person_id, tracking_images=None):
    """POST a frisking_missed event to the Node.js server."""
    if not upload_config.get('enabled', False):
        return False

    server_url = upload_config['server_url']
    endpoint = upload_config['upload_endpoint']
    camera_id = upload_config['camera_id']
    label = upload_config.get('label', 'frisking_missed')
    max_image_kb = upload_config.get('event_image_max_kb', 900)
    max_thumb_kb = upload_config.get('thumbnail_max_kb', 25)
    thumb_size = tuple(upload_config.get('thumbnail_size', [175, 175]))
    timeout = upload_config.get('upload_timeout_seconds', 15)

    url = f"{server_url}{endpoint}"
    event_payload = build_event_payload(camera_id, label, bbox_xyxy)

    image_bytes = compress_image_to_bytes(event_frame, max_kb=max_image_kb)
    if not image_bytes:
        print("  UPLOAD ERROR: Failed to compress event image")
        return False

    thumb_bytes = generate_thumbnail(event_frame, size=thumb_size, max_kb=max_thumb_kb)
    if not thumb_bytes:
        print("  UPLOAD ERROR: Failed to generate thumbnail")
        return False

    print(f"  Image: {len(image_bytes)/1024:.1f}KB | Thumbnail: {len(thumb_bytes)/1024:.1f}KB")

    files = {
        'image': ('event.jpg', image_bytes, 'image/jpeg'),
        'thumbnail': ('thumbnail.jpg', thumb_bytes, 'image/jpeg'),
    }
    data = {
        'event': json.dumps(event_payload),
        'label': label,
        'bbox': json.dumps(list(bbox_xyxy)),
    }
    if tracking_images:
        data['tracking_images'] = json.dumps(tracking_images)
        print(f"  Tracking images: {len(tracking_images)} frames")

    try:
        response = requests.post(url, files=files, data=data, timeout=timeout)
        if response.status_code == 200:
            result = response.json()
            if result.get('skipped'):
                print(f"  UPLOAD SKIPPED: {result.get('message', 'unknown reason')}")
            else:
                print(f"  UPLOAD SUCCESS: Event created for {person_id}")
            return True

        print(f"  UPLOAD FAILED: HTTP {response.status_code} - {response.text[:200]}")
        return False
    except requests.exceptions.ConnectionError:
        print(f"  UPLOAD FAILED: Cannot connect to {url}")
        return False
    except requests.exceptions.Timeout:
        print(f"  UPLOAD FAILED: Request timed out ({timeout}s)")
        return False
    except Exception as e:
        print(f"  UPLOAD FAILED: {e}")
        return False


def make_tracking_frame_record(frame, frame_count, label):
    return {
        "frame": frame.copy(),
        "frame_count": frame_count,
        "frame_time": time.time(),
        "label": label,
    }


def encode_tracking_frame(frame):
    if TRACKING_FRAME_MAX_WIDTH > 0 and frame.shape[1] > TRACKING_FRAME_MAX_WIDTH:
        scale = TRACKING_FRAME_MAX_WIDTH / frame.shape[1]
        frame = cv2.resize(
            frame,
            (TRACKING_FRAME_MAX_WIDTH, int(frame.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )

    success, buffer = cv2.imencode(
        '.jpg',
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, TRACKING_FRAME_JPEG_QUALITY]
    )
    if not success:
        return None
    return base64.b64encode(buffer).decode('ascii')


def build_tracking_images(frame_records, bbox_xyxy, default_label):
    tracking_images = []
    for record in frame_records:
        encoded_image = encode_tracking_frame(record["frame"])
        if not encoded_image:
            continue
        tracking_images.append({
            "image": encoded_image,
            "bbox": list(bbox_xyxy),
            "frame_time": record["frame_time"],
            "label": record.get("label") or default_label,
        })
    return tracking_images


def prune_tracking_history(previous_tracking_frames, now):
    cutoff = now - TRACKING_HISTORY_SECONDS
    while previous_tracking_frames and previous_tracking_frames[0]["frame_time"] < cutoff:
        previous_tracking_frames.popleft()


def sample_tracking_records(previous_tracking_frames, current_record):
    cutoff = current_record["frame_time"] - TRACKING_HISTORY_SECONDS
    records = [
        record
        for record in list(previous_tracking_frames) + [current_record]
        if record["frame_time"] >= cutoff
    ]
    if len(records) <= TRACKING_EVENT_FRAMES:
        return records

    last_index = len(records) - 1
    return [
        records[round(index * last_index / (TRACKING_EVENT_FRAMES - 1))]
        for index in range(TRACKING_EVENT_FRAMES)
    ]

class FriskingDetector:
    def __init__(self, model_path="yolo11l-pose.pt", person_model_path=None, confidence=0.5, max_frames=30, distance_threshold=100):
        self.model = YOLO(model_path)  # Pose model: keypoints/frisking hand activity
        self.pose_model_path = model_path
        env_person_model_path = getenv('PERSON_MODEL_PATH', '').strip()
        resolved_person_model_path = person_model_path if person_model_path is not None else env_person_model_path
        if isinstance(resolved_person_model_path, str):
            resolved_person_model_path = resolved_person_model_path.strip()
        if str(resolved_person_model_path).lower() in ('', '0', 'none', 'off', 'false', 'no'):
            resolved_person_model_path = None
        self.person_model_path = resolved_person_model_path
        self.person_model = YOLO(resolved_person_model_path) if resolved_person_model_path else None
        self.confidence = confidence
        self.max_frames = max_frames
        self.distance_threshold = distance_threshold  # Threshold for detecting close persons
        self.person_tracks = {}  # Dictionary to track different people
        self.interaction_history = deque(maxlen=max_frames)
        self.frisking_detected = False
        self.last_frisking_time = 0
        self.person_id_counter = 0
        self.walking_threshold = 15  # Lower threshold for movement towards camera
        self.direction_consistency_frames = 8  # Fewer frames for entry hall scenario
        self.track_max_age_frames = 45 # Increased from 10 to 45 to prevent track loss if person turns around
        self.tracking_match_threshold = 140
        self.tracker_config = getenv('YOLO_TRACKER', 'botsort.yaml')
        self.person_tracker_config = getenv('PERSON_YOLO_TRACKER', self.tracker_config)
        self.pose_person_match_iou_threshold = float(getenv('POSE_PERSON_MATCH_IOU_THRESHOLD', '0.15'))
        self.pose_person_match_center_distance = float(getenv('POSE_PERSON_MATCH_CENTER_DISTANCE_PX', '160'))
        self.track_id_aliases = {}  # raw tracker id -> stable person_id
        self.track_reid_max_age_frames = max(
            self.track_max_age_frames,
            int(getenv('TRACK_REID_MAX_AGE_FRAMES', '75'))
        )
        self.track_reid_match_distance = float(getenv('TRACK_REID_MATCH_DISTANCE', '170'))
        self.track_reid_max_area_ratio = float(getenv('TRACK_REID_MAX_AREA_RATIO', '2.75'))
        self.min_consistent_person_bbox_frames = max(
            2, int(getenv('MIN_CONSISTENT_PERSON_BBOX_FRAMES', '6'))
        )
        self.motion_tracking_enabled = getenv('MOTION_TRACKING_ENABLED', '1').lower() in ('1', 'true', 'yes', 'on')
        self.motion_prediction_max_age_frames = max(1, int(getenv('MOTION_PREDICTION_MAX_AGE_FRAMES', '12')))
        self.motion_process_noise = float(getenv('MOTION_PROCESS_NOISE', '1.0'))
        self.motion_measurement_noise = float(getenv('MOTION_MEASUREMENT_NOISE', '10.0'))
        self.motion_initial_covariance = float(getenv('MOTION_INITIAL_COVARIANCE', '100.0'))
        self.frisking_history_frames = max(1, int(getenv('FRISKING_HISTORY_FRAMES', '4')))
        self.frisking_confirm_frames = max(1, min(
            int(getenv('FRISKING_CONFIRM_FRAMES', '1')),
            self.frisking_history_frames
        ))
        self.frisking_pair_history = deque(maxlen=self.frisking_history_frames)
        self.output_dir = "missed_frisking_alerts"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Line crossing tracking
        self.missed_frisking_people = set()  # Track people who missed frisking
        self.frisked_people = set()  # Track people who have been frisked
        self.security_people = set()  # Track security personnel so they never emit missed-frisking events
        self.line_crossing_people = {}  # Track line crossing state for each person
        self.security_overlap_frames = defaultdict(int)  # Track how many frames a person overlaps with security
        self.frisking_time_threshold = float(getenv('FRISKING_TIME_SEC', '4.0')) # Configurable frisking time in seconds
        self.horizontal_line_y = None  # Recomputed from the current frame height
        self.horizontal_line_frame_shape = None
        self.frame_count = 0  # Frame counter for single frame processing
        
        # Per-person history for retrospective frisking evaluation
        # Each entry: list of dicts with keys: near_security, standing_still, pose_frisking
        self.person_security_history = defaultdict(
            lambda: deque(maxlen=max(120, int(getenv('PERSON_SECURITY_HISTORY_FRAMES', '180'))))
        )
        
        # Grace period: when the person's bbox bottom crosses the line, wait N frames before deciding.
        # This handles the case where they cross the line THEN security frisks them.
        self.pending_line_crossings = {}  # person_id -> frame_count when they crossed
        self.pending_crossing_snapshots = {}  # person_id -> crossing frame plus frozen tracking evidence
        self.line_crossing_grace_frames = max(1, int(getenv('LINE_CROSSING_GRACE_FRAMES', '25')))
        self.line_crossing_direction_frames = max(2, int(getenv('LINE_CROSSING_DIRECTION_FRAMES', '3')))
        self.line_crossing_min_direction_delta = float(getenv('LINE_CROSSING_MIN_DIRECTION_DELTA_PX', '6'))
        self.line_crossing_max_jump_px = float(getenv('LINE_CROSSING_MAX_JUMP_PX', '350'))
        self.line_crossing_hysteresis_px = max(
            1.0,
            float(getenv('LINE_CROSSING_HYSTERESIS_PX', '24'))
        )
        self.line_crossing_source_frames = max(
            1,
            int(getenv('LINE_CROSSING_SOURCE_FRAMES', '2'))
        )
        self.line_crossing_destination_frames = max(
            1,
            int(getenv('LINE_CROSSING_DESTINATION_FRAMES', '2'))
        )
        self.line_crossing_min_total_delta = max(
            self.line_crossing_hysteresis_px * 2,
            float(getenv('LINE_CROSSING_MIN_TOTAL_DELTA_PX', '48'))
        )
        self.line_crossing_min_consistency = min(
            1.0,
            max(0.5, float(getenv('LINE_CROSSING_MIN_CONSISTENCY', '0.75')))
        )
        self.line_crossing_min_body_consistency = min(
            1.0,
            max(0.5, float(getenv('LINE_CROSSING_MIN_BODY_CONSISTENCY', '0.60')))
        )
        self.line_crossing_min_center_delta = max(
            1.0,
            float(getenv('LINE_CROSSING_MIN_CENTER_DELTA_PX', '20'))
        )
        self.line_crossing_min_top_delta = max(
            1.0,
            float(getenv('LINE_CROSSING_MIN_TOP_DELTA_PX', '10'))
        )
        self.line_crossing_max_transition_frames = max(
            self.line_crossing_source_frames + self.line_crossing_destination_frames,
            int(getenv('LINE_CROSSING_MAX_TRANSITION_FRAMES', '90'))
        )
        self.max_line_crossing_grace_frames = max(
            self.line_crossing_grace_frames,
            int(getenv('MAX_LINE_CROSSING_GRACE_FRAMES', '120'))
        )
        self.grace_extend_near_security_frames = max(
            1,
            int(getenv('GRACE_EXTEND_NEAR_SECURITY_FRAMES', '2'))
        )
        self.pre_crossing_pose_frames = max(1, int(getenv('PRE_CROSSING_POSE_FRAMES', '1')))
        self.min_frisking_pose_frames = max(1, int(getenv('MIN_FRISKING_POSE_FRAMES', '1')))
        self.min_near_still_frames = max(1, int(getenv('MIN_NEAR_STILL_FRAMES', '4')))
        self.frisking_reach_score_threshold = max(1, int(getenv('FRISKING_REACH_SCORE_THRESHOLD', '3')))
        self.frisking_score_threshold = max(1, int(getenv(
            'FRISKING_SCORE_THRESHOLD',
            str(max(self.min_near_still_frames * 2, self.min_frisking_pose_frames * 4))
        )))
        self.recent_interaction_window = max(
            self.min_frisking_pose_frames,
            self.min_near_still_frames,
            int(getenv('RECENT_INTERACTION_WINDOW_FRAMES', '60'))
        )
        self.strong_pose_latch_frames = max(
            1,
            int(getenv('FRISKING_STRONG_POSE_LATCH_FRAMES', str(self.min_frisking_pose_frames)))
        )
        self.frisking_strong_reach_score_threshold = max(
            self.frisking_reach_score_threshold,
            int(getenv('FRISKING_STRONG_REACH_SCORE_THRESHOLD', str(self.frisking_reach_score_threshold)))
        )
        self.quick_frisking_enabled = getenv('QUICK_FRISKING_ENABLED', '1').lower() in ('1', 'true', 'yes', 'on')
        self.quick_frisking_contact_score_threshold = max(
            1,
            int(getenv('QUICK_FRISKING_CONTACT_SCORE_THRESHOLD', str(self.frisking_reach_score_threshold)))
        )
        self.quick_frisking_contact_radius_px = float(getenv('QUICK_FRISKING_CONTACT_RADIUS_PX', '70'))
        self.security_confirm_frames = max(1, int(getenv('SECURITY_CONFIRM_FRAMES', '3')))
        self.security_missed_skip_frames = max(1, int(getenv('SECURITY_MISSED_SKIP_FRAMES', '2')))
        self.min_smooth_track_frames = max(2, int(getenv('MIN_SMOOTH_TRACK_FRAMES', '4')))
        self.smooth_no_security_grace_frames = max(
            1,
            min(
                self.line_crossing_grace_frames,
                int(getenv('SMOOTH_NO_SECURITY_GRACE_FRAMES', '3'))
            )
        )
        self.pending_crossing_grace_frames = {}
        self.debug_frisking = getenv('FRISKING_DEBUG', '0').lower() in ('1', 'true', 'yes', 'on')
        self.debug_log_every_frames = max(1, int(getenv('FRISKING_DEBUG_EVERY_FRAMES', '15')))
        self.debug_label = 'out'
        self.debug_log(
            (
                "crossing direction=BOTTOM->TOP "
                f"hysteresis={self.line_crossing_hysteresis_px:.1f}px "
                f"source_frames={self.line_crossing_source_frames} "
                f"destination_frames={self.line_crossing_destination_frames} "
                f"min_delta={self.line_crossing_min_total_delta:.1f}px "
                f"consistency={self.line_crossing_min_consistency:.2f}"
            ),
            force=True
        )
        self.debug_log(
            (
                f"hybrid motion tracking={'ON' if self.motion_tracking_enabled else 'OFF'} "
                f"kalman_prediction_age={self.motion_prediction_max_age_frames} frames "
                f"reid_gate={self.track_reid_match_distance:.1f}px"
            ),
            force=True
        )

    def process_frame(self, frame):
        results = self.model.track(frame, conf=self.confidence, stream=True, persist=True, tracker=self.tracker_config)
        return results

    def process_person_frame(self, frame):
        if self.person_model is None:
            return None
        return self.person_model.track(
            frame,
            conf=self.confidence,
            classes=[0],
            stream=True,
            persist=True,
            tracker=self.person_tracker_config
        )

    def debug_log(self, message, person_id=None, force=False):
        if not self.debug_frisking:
            return
        if not force and self.frame_count % self.debug_log_every_frames != 0:
            return

        if person_id:
            prefix = f"[frisking:{self.debug_label}:{person_id}]"
        else:
            prefix = f"[frisking:{self.debug_label}]"
        print(f"{prefix} f={self.frame_count} {message}")

    def evidence_counts(self, person_id, max_frames=None):
        history = list(self.person_security_history.get(person_id, []))
        if max_frames is not None:
            history = history[-max_frames:]

        frisking_scores = [int(h.get('frisking_score', 0) or 0) for h in history]
        reach_scores = [int(h.get('reach_score', 0) or 0) for h in history]

        return {
            'near_security': sum(1 for h in history if h.get('near_security')),
            'near_still': sum(
                1 for h in history
                if h.get('near_security') and h.get('standing_still')
            ),
            'pose_frisking': sum(1 for h in history if h.get('pose_frisking')),
            'frisking_score': sum(frisking_scores),
            'max_reach_score': max(reach_scores) if reach_scores else 0,
        }

    def frame_frisking_score(self, near_security, standing_still, pose_frisking, reach_score):
        score = 0
        if near_security and standing_still:
            score += 2
        if pose_frisking:
            score += max(4, int(reach_score or 0))
        return score

    def is_frisking_confirmed(self, person_id, max_frames=None):
        counts = self.evidence_counts(person_id, max_frames)
        has_required_pattern = (
            counts['pose_frisking'] >= self.min_frisking_pose_frames
            or counts['near_still'] >= self.min_near_still_frames
        )
        return counts['frisking_score'] >= self.frisking_score_threshold and has_required_pattern

    def has_recent_frisking_interaction(self, person_id):
        if self.is_frisking_confirmed(person_id, self.recent_interaction_window):
            return True

        counts = self.evidence_counts(person_id, self.recent_interaction_window)
        return (
            counts['pose_frisking'] >= self.strong_pose_latch_frames
            and counts['max_reach_score'] >= self.frisking_strong_reach_score_threshold
        )

    def mark_frisked_if_confirmed(self, person_id, reason):
        if person_id in self.security_people:
            return False

        if person_id in self.frisked_people:
            return True

        if not self.has_recent_frisking_interaction(person_id):
            return False

        counts = self.evidence_counts(person_id, self.recent_interaction_window)
        self.frisked_people.add(person_id)
        self.missed_frisking_people.discard(person_id)
        self.debug_log(
            (
                f"FRISKED latch {reason} "
                f"recent_pose={counts['pose_frisking']}/{self.strong_pose_latch_frames} "
                f"recent_near_still={counts['near_still']}/{self.min_near_still_frames} "
                f"recent_score={counts['frisking_score']}/{self.frisking_score_threshold} "
                f"max_reach={counts['max_reach_score']}/{self.frisking_strong_reach_score_threshold}"
            ),
            person_id,
            force=True
        )
        return True

    def mark_security_person(self, person_id, reason):
        self.security_people.add(person_id)
        self.frisked_people.discard(person_id)
        self.missed_frisking_people.discard(person_id)
        self.pending_line_crossings.pop(person_id, None)
        self.pending_crossing_grace_frames.pop(person_id, None)
        self.pending_crossing_snapshots.pop(person_id, None)
        self.line_crossing_people.pop(person_id, None)
        self.debug_log(reason, person_id, force=True)

    def should_extend_pending_grace(self, person_id, frames_since_crossing):
        if frames_since_crossing >= self.max_line_crossing_grace_frames:
            return False

        counts = self.evidence_counts(person_id, self.recent_interaction_window)
        return (
            counts['pose_frisking'] > 0
            or counts['frisking_score'] > 0
        )

    def has_no_security_contact(self, person_id):
        counts = self.evidence_counts(person_id, self.recent_interaction_window)
        return (
            counts['pose_frisking'] == 0
            and counts['frisking_score'] == 0
            and counts['max_reach_score'] == 0
        )

    def crossing_grace_for_track(self, person_id, track):
        smooth_frames = track.get('consecutive_detected_frames', 0)
        if smooth_frames >= self.min_smooth_track_frames and self.has_no_security_contact(person_id):
            return self.smooth_no_security_grace_frames
        return self.line_crossing_grace_frames

    def pending_grace_for_person(self, person_id):
        return self.pending_crossing_grace_frames.get(
            person_id,
            self.line_crossing_grace_frames
        )

    def save_frame(self, frame, filename="frisking_detected.jpg"):
        print(f"Frisking detected (visualization only)")
        return filename
    
    def save_missed_frisking_frame(self, frame, bbox, person_id):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/missed_frisking_{person_id}_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Missed frisking frame saved as {filename}")
        return filename
    
    def save_line_crossing_event(self, frame, person_id):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/missed_frisking_{person_id}_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Missed frisking event saved as {filename}")
        return filename

    def save_previous_track_event(self, frame, person_id, frame_count):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/missed_frisking_track_{person_id}_{frame_count}_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Previous missed frisking track frame saved as {filename}")
        return filename

    def bbox_xywh_to_xyxy(self, bbox):
        x, y, w, h = bbox
        return [int(x), int(y), int(x + w), int(y + h)]

    def build_previous_track_events(self, person_id, track, final_frame_count, snapshots=None):
        previous_events = []
        snapshot_source = snapshots if snapshots is not None else track.get('event_snapshots')
        for snapshot in list(snapshot_source or []):
            snapshot_frame_count = snapshot.get('frame_count')
            snapshot_person_id = snapshot.get('person_id')
            if snapshot_person_id is not None and snapshot_person_id != person_id:
                continue
            if snapshot_frame_count is None:
                continue
            if snapshots is None and snapshot_frame_count >= final_frame_count:
                continue
            if snapshots is not None and snapshot_frame_count > final_frame_count:
                continue

            bbox = snapshot.get('tracking_bbox') or snapshot.get('bbox')
            frame = snapshot.get('frame')
            if frame is None or not bbox or bbox[2] <= 10 or bbox[3] <= 10:
                continue

            clean_frame = self.draw_horizontal_line(frame.copy())
            clean_frame = self.draw_bbox_with_label(
                clean_frame, bbox, PREVIOUS_FRISKING_TRACK_EVENT_LABEL,
                color=(0, 0, 255), thickness=3
            )
            filename = self.save_previous_track_event(clean_frame, person_id, snapshot_frame_count)
            previous_events.append({
                "person_id": person_id,
                "bbox": bbox,
                "bbox_xyxy": self.bbox_xywh_to_xyxy(bbox),
                "frame_count": snapshot_frame_count,
                "frame_time": snapshot.get('frame_time', time.time()),
                "label": PREVIOUS_FRISKING_TRACK_EVENT_LABEL,
                "filename": filename,
                "frame": clean_frame
            })

        return previous_events
    
    def draw_horizontal_line(self, frame):
        frame_shape = frame.shape[:2]
        if self.horizontal_line_y is None or self.horizontal_line_frame_shape != frame_shape:
            self.horizontal_line_y = int(frame.shape[0] * 0.50)  # Middle of the frame
            self.horizontal_line_frame_shape = frame_shape

        line_y = int(np.clip(self.horizontal_line_y, 0, frame.shape[0] - 1))
        
        # cv2.line(frame, (0, line_y), (frame.shape[1] - 1, line_y), (0, 0, 255), 2)
        width = frame.shape[1]

        start_x = int(width * 0.30)   # 30% from left
        end_x   = int(width * 0.70)   # 70% from left

        cv2.line(frame, (start_x, line_y), (end_x, line_y), (0, 0, 255), 2)
        
        label_y = max(line_y - 10, 20)
        cv2.putText(frame, "FRISKING LINE (ENTRY)", (10, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # Draw upward arrow to show detection direction
        arrow_x = frame.shape[1] - 60
        cv2.arrowedLine(frame, (arrow_x, line_y + 30), (arrow_x, line_y - 30), (0, 0, 255), 3, tipLength=0.4)
        
        return frame

    def new_line_crossing_state(self, current_y):
        return {
            'last_y': float(current_y),
            'has_crossed': False,
            'armed': False,
            'source_frames': 0,
            'destination_frames': 0,
            'source_y': None,
            'candidate_started': None,
            'samples': [],
            'center_samples': [],
            'top_samples': [],
            'boundary_snapshot': None
        }

    def reset_line_crossing_candidate(self, crossing_data, current_y):
        crossing_data.update({
            'last_y': float(current_y),
            'armed': False,
            'source_frames': 0,
            'destination_frames': 0,
            'source_y': None,
            'candidate_started': None,
            'samples': [],
            'center_samples': [],
            'top_samples': [],
            'boundary_snapshot': None
        })

    def upward_motion_metrics(self, samples):
        if len(samples) < 2:
            return 0.0, 0.0

        total_delta = float(samples[0]) - float(samples[-1])
        expected_distance = 0.0
        opposite_distance = 0.0
        for previous_y, next_y in zip(samples, samples[1:]):
            step = float(next_y) - float(previous_y)
            if step < 0:
                expected_distance += -step
            elif step > 0:
                opposite_distance += step

        travelled = expected_distance + opposite_distance
        consistency = expected_distance / travelled if travelled > 0 else 0.0
        return total_delta, consistency

    def line_crossing_direction_ok(self, foot_samples, center_samples, top_samples):
        foot_delta, foot_consistency = self.upward_motion_metrics(foot_samples)
        center_delta, center_consistency = self.upward_motion_metrics(center_samples)
        top_delta, top_consistency = self.upward_motion_metrics(top_samples)
        direction_ok = (
            foot_delta >= self.line_crossing_min_total_delta
            and foot_consistency >= self.line_crossing_min_consistency
            and center_delta >= self.line_crossing_min_center_delta
            and center_consistency >= self.line_crossing_min_body_consistency
            and top_delta >= self.line_crossing_min_top_delta
            and top_consistency >= self.line_crossing_min_body_consistency
        )
        return direction_ok, {
            'foot_delta': foot_delta,
            'foot_consistency': foot_consistency,
            'center_delta': center_delta,
            'center_consistency': center_consistency,
            'top_delta': top_delta,
            'top_consistency': top_consistency,
        }

    def check_line_crossing(self, person_id, current_y, bbox=None):
        """Confirm an outgate crossing only after a full bottom-to-top traversal."""
        line_y = float(self.horizontal_line_y)
        current_y = float(current_y)
        if not bbox or len(bbox) < 4:
            return False
        current_top_y = float(bbox[1])
        current_center_y = float(bbox[1] + bbox[3] / 2.0)
        source_limit = line_y + self.line_crossing_hysteresis_px
        destination_limit = line_y - self.line_crossing_hysteresis_px
        on_source_side = current_y >= source_limit
        on_destination_side = current_y <= destination_limit
        track = self.person_tracks.get(person_id)
        if track is not None:
            origin_side = track.get('line_origin_side')
            if origin_side is None and (on_source_side or on_destination_side):
                origin_side = 'source' if on_source_side else 'destination'
                track['line_origin_side'] = origin_side
                self.debug_log(
                    f"line origin locked to {origin_side.upper()} side",
                    person_id,
                    force=True
                )
            if origin_side == 'destination':
                return False

        crossing_data = self.line_crossing_people.setdefault(
            person_id,
            self.new_line_crossing_state(current_y)
        )

        if crossing_data.get('has_crossed'):
            crossing_data['last_y'] = current_y
            return False

        if on_source_side:
            was_armed = crossing_data.get('armed', False)
            crossing_data['source_frames'] = crossing_data.get('source_frames', 0) + 1
            crossing_data['destination_frames'] = 0
            crossing_data['source_y'] = current_y
            crossing_data['candidate_started'] = None
            crossing_data['samples'] = [current_y]
            crossing_data['center_samples'] = [current_center_y]
            crossing_data['top_samples'] = [current_top_y]
            crossing_data['boundary_snapshot'] = None
            crossing_data['armed'] = (
                crossing_data['source_frames'] >= self.line_crossing_source_frames
            )
            crossing_data['last_y'] = current_y
            if crossing_data['armed'] and not was_armed:
                self.debug_log(
                    (
                        "crossing armed on BOTTOM side "
                        f"foot_y={current_y:.1f} source_limit={source_limit:.1f}"
                    ),
                    person_id,
                    force=True
                )
            return False

        if not crossing_data.get('armed'):
            crossing_data['source_frames'] = 0
            crossing_data['last_y'] = current_y
            return False

        if crossing_data.get('candidate_started') is None:
            crossing_data['candidate_started'] = self.frame_count

        previous_y = float(crossing_data.get('last_y', current_y))
        crossing_data.setdefault('samples', []).append(current_y)
        crossing_data.setdefault('center_samples', []).append(current_center_y)
        crossing_data.setdefault('top_samples', []).append(current_top_y)
        if previous_y >= line_y and current_y < line_y:
            self.remember_line_boundary_crossing_snapshot(
                person_id, crossing_data, track, self.frame_count
            )
        elif crossing_data.get('boundary_snapshot') is not None:
            self.remember_boundary_post_crossing_snapshot(
                crossing_data, track, self.frame_count
            )
        transition_age = self.frame_count - crossing_data['candidate_started']
        if transition_age > self.line_crossing_max_transition_frames:
            self.debug_log(
                f"crossing candidate expired after {transition_age} frames",
                person_id,
                force=True
            )
            self.reset_line_crossing_candidate(crossing_data, current_y)
            return False

        if on_destination_side:
            crossing_data['destination_frames'] += 1
        else:
            crossing_data['destination_frames'] = 0
        crossing_data['last_y'] = current_y

        if crossing_data['destination_frames'] < self.line_crossing_destination_frames:
            return False

        direction_ok, metrics = self.line_crossing_direction_ok(
            crossing_data['samples'],
            crossing_data['center_samples'],
            crossing_data['top_samples']
        )
        if not direction_ok:
            self.debug_log(
                (
                    "crossing rejected: incomplete/wrong-direction trajectory "
                    f"foot={metrics['foot_delta']:.1f}/{self.line_crossing_min_total_delta:.1f} "
                    f"center={metrics['center_delta']:.1f}/{self.line_crossing_min_center_delta:.1f} "
                    f"top={metrics['top_delta']:.1f}/{self.line_crossing_min_top_delta:.1f} "
                    f"body_consistency={metrics['center_consistency']:.2f},"
                    f"{metrics['top_consistency']:.2f}/{self.line_crossing_min_body_consistency:.2f}"
                ),
                person_id,
                force=True
            )
            self.reset_line_crossing_candidate(crossing_data, current_y)
            return False

        crossing_data['has_crossed'] = True
        self.debug_log(
            (
                "CONFIRMED BOTTOM->TOP crossing "
                f"foot_delta={metrics['foot_delta']:.1f} "
                f"center_delta={metrics['center_delta']:.1f} "
                f"top_delta={metrics['top_delta']:.1f} "
                f"destination_frames={crossing_data['destination_frames']}"
            ),
            person_id,
            force=True
        )
        return True

    def is_valid_keypoint(self, point):
        if point is None or len(point) < 2:
            return False
        x, y = point[:2]
        return np.isfinite(x) and np.isfinite(y) and not (abs(x) < 1 and abs(y) < 1)

    def mean_valid_keypoints(self, keypoints, indices):
        points = [keypoints[idx][:2] for idx in indices if idx < len(keypoints) and self.is_valid_keypoint(keypoints[idx])]
        if not points:
            return None
        return np.mean(points, axis=0)

    def valid_keypoints_for_indices(self, keypoints, indices):
        return [keypoints[idx][:2] for idx in indices if idx < len(keypoints) and self.is_valid_keypoint(keypoints[idx])]

    def keypoint_body_height(self, keypoints):
        valid_points = np.array([point[:2] for point in keypoints if self.is_valid_keypoint(point)])
        if len(valid_points) < 4:
            return 0
        return float(max(1, np.max(valid_points[:, 1]) - np.min(valid_points[:, 1])))

    def min_distance_between_points(self, points_a, points_b):
        if not points_a or not points_b:
            return float('inf')
        return min(float(np.linalg.norm(np.array(a) - np.array(b))) for a in points_a for b in points_b)

    def is_reaching_towards_person(self, actor_kp, subject_kp, actor_height, subject_height):
        actor_torso = self.mean_valid_keypoints(actor_kp, [5, 6, 11, 12])
        subject_torso = self.mean_valid_keypoints(subject_kp, [5, 6, 11, 12])
        subject_upper_body = self.valid_keypoints_for_indices(subject_kp, [5, 6, 11, 12])
        actor_wrists = self.valid_keypoints_for_indices(actor_kp, [9, 10])
        actor_elbows = self.valid_keypoints_for_indices(actor_kp, [7, 8])
        actor_hips = self.valid_keypoints_for_indices(actor_kp, [11, 12])

        if actor_torso is None or subject_torso is None or not actor_wrists or not actor_hips:
            return False, 0
            
        # STRICT CHECK: Are the guard's wrists raised?
        # In image coordinates, smaller Y is higher up.
        mean_hip_y = np.mean([hip[1] for hip in actor_hips])
        min_wrist_y = np.min([wrist[1] for wrist in actor_wrists])
        if min_wrist_y >= mean_hip_y - (actor_height * 0.05):
            return False, 0

        contact_radius = max(55, subject_height * 0.22)
        elbow_radius = max(75, subject_height * 0.30)
        own_body_clearance = max(35, actor_height * 0.16)

        wrist_to_subject = self.min_distance_between_points(actor_wrists, subject_upper_body)
        elbow_to_subject = self.min_distance_between_points(actor_elbows, subject_upper_body)
        wrist_to_own_torso = min(float(np.linalg.norm(np.array(wrist) - actor_torso)) for wrist in actor_wrists)
        wrist_to_subject_torso = min(float(np.linalg.norm(np.array(wrist) - subject_torso)) for wrist in actor_wrists)

        wrist_contact = wrist_to_subject <= contact_radius
        elbow_support = elbow_to_subject <= elbow_radius
        hand_is_extended = wrist_to_own_torso >= own_body_clearance
        hand_prefers_subject = wrist_to_subject_torso < wrist_to_own_torso

        # Strict requirement: The guard's wrist MUST be reaching out towards the subject
        # meaning it is physically closer to the subject's torso than their own torso.
        # This absolutely prevents false positives from arms crossed or resting at sides.
        if not hand_prefers_subject:
            return False, 0

        score = 0
        if wrist_contact:
            score += 2
        if elbow_support:
            score += 1
        if hand_is_extended:
            score += 1

        return wrist_contact and hand_is_extended, score

    def is_quick_frisking_contact(self, actor_kp, subject_kp, actor_height, subject_height):
        if not self.quick_frisking_enabled:
            return False, 0

        actor_torso = self.mean_valid_keypoints(actor_kp, [5, 6, 11, 12])
        subject_torso = self.mean_valid_keypoints(subject_kp, [5, 6, 11, 12])
        subject_contact_points = self.valid_keypoints_for_indices(subject_kp, [5, 6, 7, 8, 11, 12])
        actor_wrists = self.valid_keypoints_for_indices(actor_kp, [9, 10])
        actor_elbows = self.valid_keypoints_for_indices(actor_kp, [7, 8])

        if actor_torso is None or subject_torso is None or not actor_wrists or not subject_contact_points:
            return False, 0

        torso_distance = float(np.linalg.norm(actor_torso - subject_torso))
        proximity_limit = max(self.distance_threshold, min(230, ((actor_height + subject_height) / 2.0) * 0.95))
        if torso_distance > proximity_limit:
            return False, 0

        contact_radius = max(self.quick_frisking_contact_radius_px, subject_height * 0.28)
        elbow_radius = max(self.quick_frisking_contact_radius_px + 20, subject_height * 0.36)

        wrist_to_subject = self.min_distance_between_points(actor_wrists, subject_contact_points)
        elbow_to_subject = self.min_distance_between_points(actor_elbows, subject_contact_points)
        wrist_to_own_torso = min(float(np.linalg.norm(np.array(wrist) - actor_torso)) for wrist in actor_wrists)
        wrist_to_subject_torso = min(float(np.linalg.norm(np.array(wrist) - subject_torso)) for wrist in actor_wrists)

        wrist_contact = wrist_to_subject <= contact_radius
        elbow_support = elbow_to_subject <= elbow_radius
        hand_is_extended = wrist_to_own_torso >= max(25, actor_height * 0.10)
        hand_prefers_subject = wrist_to_subject_torso <= wrist_to_own_torso * 1.15
        close_pair = torso_distance <= proximity_limit * 0.70

        score = 0
        if wrist_contact:
            score += 2
        if elbow_support:
            score += 1
        if hand_is_extended:
            score += 1
        if hand_prefers_subject:
            score += 1
        if close_pair:
            score += 1

        confirmed = (
            wrist_contact
            and score >= self.quick_frisking_contact_score_threshold
            and (hand_prefers_subject or hand_is_extended or elbow_support)
        )
        return confirmed, score

    def is_frisking_pair_candidate(self, kp1, kp2):
        torso_kp1 = self.mean_valid_keypoints(kp1, [5, 6, 11, 12])
        torso_kp2 = self.mean_valid_keypoints(kp2, [5, 6, 11, 12])
        if torso_kp1 is None or torso_kp2 is None:
            return False

        height1 = self.keypoint_body_height(kp1)
        height2 = self.keypoint_body_height(kp2)
        if height1 < 40 or height2 < 40:
            return False

        torso_distance = float(np.linalg.norm(torso_kp1 - torso_kp2))
        proximity_threshold = max(self.distance_threshold, min(180, ((height1 + height2) / 2) * 0.75))
        if torso_distance > proximity_threshold:
            return False

        vertical_gap = abs(float(torso_kp1[1] - torso_kp2[1]))
        if vertical_gap > max(height1, height2) * 0.45:
            return False

        reach_1_to_2, score_1_to_2 = self.is_reaching_towards_person(kp1, kp2, height1, height2)
        reach_2_to_1, score_2_to_1 = self.is_reaching_towards_person(kp2, kp1, height2, height1)
        best_score = max(score_1_to_2, score_2_to_1)

        return (reach_1_to_2 or reach_2_to_1) and best_score >= 4

    def is_frisking_detected(self, keypoints_list):
        if len(keypoints_list) < 2:
            self.frisking_pair_history.append(set())
            return False, []
        
        candidate_pairs = []
        
        for i in range(len(keypoints_list)):
            for j in range(i + 1, len(keypoints_list)):
                kp1 = keypoints_list[i]
                kp2 = keypoints_list[j]

                if self.is_frisking_pair_candidate(kp1, kp2):
                    candidate_pairs.append((i, j))

        candidate_pair_set = set(candidate_pairs)
        self.frisking_pair_history.append(candidate_pair_set)

        confirmed_pairs = []
        for pair in candidate_pairs:
            pair_hits = sum(1 for frame_pairs in self.frisking_pair_history if pair in frame_pairs)
            if pair_hits >= self.frisking_confirm_frames:
                confirmed_pairs.append(pair)

        if confirmed_pairs:
            return True, confirmed_pairs

        return False, []
    
    def calculate_bbox_from_keypoints(self, keypoints):
        if len(keypoints) == 0:
            return None
        
        # Filter out (0,0) which represent undetected keypoints
        valid_points = keypoints[(keypoints[:, 0] > 0) | (keypoints[:, 1] > 0)]
        if len(valid_points) == 0:
            return None
            
        x_min = int(np.min(valid_points[:, 0]))
        y_min = int(np.min(valid_points[:, 1]))
        x_max = int(np.max(valid_points[:, 0]))
        y_max = int(np.max(valid_points[:, 1]))
        
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = x_max + padding
        y_max = y_max + padding
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def is_walking_straight(self, person_id):
        if person_id not in self.person_tracks:
            return False
        
        track = self.person_tracks[person_id]
        positions = track.get('positions', [])
        bbox_history = track.get('bbox_history', [])
        
        if len(positions) < self.direction_consistency_frames:
            return False
        
        positions_list = list(positions)
        recent_positions = positions_list[-self.direction_consistency_frames:]
        
        movements = []
        for i in range(1, len(recent_positions)):
            dx = recent_positions[i][0] - recent_positions[i-1][0]
            dy = recent_positions[i][1] - recent_positions[i-1][1]
            movement_magnitude = math.sqrt(dx**2 + dy**2)
            if movement_magnitude > self.walking_threshold:
                movements.append((dx, dy))
        
        is_approaching = False
        if len(bbox_history) >= 3:
            bbox_list = list(bbox_history)[-3:]
            sizes = [bbox[2] * bbox[3] for bbox in bbox_list]
            if len(sizes) >= 2 and sizes[-1] > sizes[0] * 1.1:
                is_approaching = True
        
        if len(movements) < 2:
            return is_approaching
        
        direction_variance = np.var([math.atan2(m[1], m[0]) for m in movements])
        walking_detected = direction_variance < 1.0 and len(movements) >= 2
        
        return walking_detected or is_approaching
    
    def is_person_near_others(self, person_id, keypoints_list):
        if person_id not in self.person_tracks:
            return False
        
        track = self.person_tracks[person_id]
        positions = track.get('positions', [])
        
        if len(positions) == 0:
            return False
        
        current_pos = positions[-1]
        
        for other_id, other_track in self.person_tracks.items():
            if other_id == person_id:
                continue
                
            other_positions = other_track.get('positions', [])
            if len(other_positions) == 0:
                continue
                
            other_pos = other_positions[-1]
            distance = math.sqrt((current_pos[0] - other_pos[0])**2 + (current_pos[1] - other_pos[1])**2)
            
            if distance < self.distance_threshold * 1.5:
                return True
        
        return False

    def create_track_state(self, frame_count):
        return {
            'positions': deque(maxlen=50),
            'first_seen': frame_count,
            'last_seen': frame_count,
            'keypoints': None,
            'keypoints_frame_count': None,
            'foot_y_history': deque(maxlen=max(6, self.line_crossing_direction_frames + 2)),
            'bbox_history': deque(maxlen=10),
            'event_snapshots': deque(maxlen=max(2, TRACKING_EVENT_FRAMES)),
            'motion_tracker': None,
            'predicted_bbox': None,
            'motion_prediction_frame': None,
            'motion_prediction_age': 0,
            'motion_prediction_only': False,
            'consecutive_detected_frames': 0,
            'consecutive_person_bbox_frames': 0,
            'last_person_bbox_frame': None,
            'pose_missing_frames': 0,
            'line_origin_side': None,
            'tracker_ids': set()
        }

    def prepare_motion_predictions(self, frame_count):
        for track in self.person_tracks.values():
            track['motion_prediction_only'] = False
            age = frame_count - track.get('last_seen', frame_count)
            tracker = track.get('motion_tracker')
            if not self.motion_tracking_enabled or tracker is None or age <= 0:
                continue
            if age > self.motion_prediction_max_age_frames:
                track['predicted_bbox'] = None
                track['motion_prediction_frame'] = None
                continue

            try:
                track['predicted_bbox'] = tracker.predict(frame_count)
                track['motion_prediction_frame'] = frame_count
                track['motion_prediction_age'] = age
                track['motion_prediction_only'] = True
            except Exception as exc:
                track['predicted_bbox'] = None
                track['motion_prediction_frame'] = None
                self.debug_log(f"motion prediction reset after error: {exc}")

    def correct_track_motion(self, track, bbox, frame_count):
        if not self.motion_tracking_enabled:
            return

        tracker = track.get('motion_tracker')
        try:
            if tracker is None:
                tracker = KalmanBBoxTracker(
                    bbox,
                    frame_count,
                    process_noise=self.motion_process_noise,
                    measurement_noise=self.motion_measurement_noise,
                    initial_covariance=self.motion_initial_covariance,
                )
                track['motion_tracker'] = tracker
                corrected_bbox = tuple(float(value) for value in bbox)
            else:
                corrected_bbox = tracker.update(bbox, frame_count)
        except Exception as exc:
            self.debug_log(f"motion correction reinitialized after error: {exc}")
            tracker = KalmanBBoxTracker(bbox, frame_count)
            track['motion_tracker'] = tracker
            corrected_bbox = tuple(float(value) for value in bbox)

        track['predicted_bbox'] = corrected_bbox
        track['motion_prediction_frame'] = frame_count
        track['motion_prediction_age'] = 0
        track['motion_prediction_only'] = False

    def current_predicted_bbox(self, track, frame_count=None):
        expected_frame = self.frame_count if frame_count is None else frame_count
        if (
            track.get('motion_prediction_only')
            and track.get('motion_prediction_frame') == expected_frame
            and track.get('motion_prediction_age', 0) <= self.motion_prediction_max_age_frames
        ):
            return track.get('predicted_bbox')
        return None

    def track_visible_this_frame(self, track, frame_count):
        return (
            track.get('last_seen') == frame_count
            or self.current_predicted_bbox(track, frame_count) is not None
        )

    def bbox_area(self, bbox):
        if not bbox:
            return 0
        return max(1, bbox[2]) * max(1, bbox[3])

    def bbox_center(self, bbox):
        if not bbox:
            return None
        return bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2

    def normalize_bbox_xyxy(self, xyxy):
        if xyxy is None or len(xyxy) < 4:
            return None

        x1, y1, x2, y2 = [float(value) for value in xyxy[:4]]
        left = int(max(0, min(x1, x2)))
        top = int(max(0, min(y1, y2)))
        right = int(max(left + 1, max(x1, x2)))
        bottom = int(max(top + 1, max(y1, y2)))
        return left, top, right - left, bottom - top

    def current_keypoints_for_track(self, track, frame_count):
        if track.get('keypoints_frame_count') != frame_count:
            return None
        return track.get('keypoints')

    def track_last_center(self, track):
        predicted_bbox = track.get('predicted_bbox')
        if track.get('motion_prediction_frame') == self.frame_count and predicted_bbox:
            return self.bbox_center(predicted_bbox)

        positions = track.get('positions', [])
        if positions:
            return positions[-1]

        bbox = self.latest_track_bbox(track)
        if bbox:
            return bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
        return None

    def stable_person_id_for_track(self, raw_track_id, current_center, current_bbox, frame_count, assigned_person_ids):
        raw_track_key = str(raw_track_id)
        existing_person_id = self.track_id_aliases.get(raw_track_key)
        if existing_person_id in self.person_tracks and existing_person_id not in assigned_person_ids:
            existing_track = self.person_tracks[existing_person_id]
            existing_center = self.track_last_center(existing_track)
            existing_bbox = self.latest_track_bbox(existing_track)
            center_ok = existing_center is not None and math.hypot(
                current_center[0] - existing_center[0],
                current_center[1] - existing_center[1]
            ) <= self.track_reid_match_distance
            existing_area = self.bbox_area(existing_bbox)
            current_area = self.bbox_area(current_bbox)
            area_ok = not (existing_area and current_area) or (
                max(existing_area, current_area) / max(1, min(existing_area, current_area))
                <= self.track_reid_max_area_ratio
            )
            if center_ok and area_ok:
                return existing_person_id

            self.track_id_aliases.pop(raw_track_key, None)
            self.debug_log(
                f"rejected stale tracker alias raw_id={raw_track_id}",
                existing_person_id,
                force=True
            )

        best_person_id = None
        best_distance = self.track_reid_match_distance
        current_area = self.bbox_area(current_bbox)

        for person_id, track in self.person_tracks.items():
            if person_id in assigned_person_ids:
                continue

            frames_missing = frame_count - track.get('last_seen', frame_count)
            if frames_missing <= 0 or frames_missing > self.track_reid_max_age_frames:
                continue

            previous_center = self.track_last_center(track)
            if previous_center is None:
                continue

            previous_bbox = self.latest_track_bbox(track)
            previous_area = self.bbox_area(previous_bbox)
            if current_area and previous_area:
                area_ratio = max(current_area, previous_area) / max(1, min(current_area, previous_area))
                if area_ratio > self.track_reid_max_area_ratio:
                    continue

            distance = math.hypot(current_center[0] - previous_center[0], current_center[1] - previous_center[1])
            if distance < best_distance:
                best_distance = distance
                best_person_id = person_id

        if best_person_id is None:
            base_person_id = f"person_{raw_track_id}"
            best_person_id = base_person_id
            while best_person_id in self.person_tracks:
                self.person_id_counter += 1
                best_person_id = f"{base_person_id}_{self.person_id_counter}"
            self.person_tracks[best_person_id] = self.create_track_state(frame_count)
        else:
            print(f"Recovered stable track {best_person_id} from tracker id {raw_track_id}")
            self.debug_log(
                f"recovered stable track from tracker id {raw_track_id}",
                best_person_id,
                force=True
            )

        self.track_id_aliases[raw_track_key] = best_person_id
        return best_person_id
    
    def extract_pose_detections(self, results):
        detections = []

        for result in results:
            if result.keypoints is None or result.boxes is None or result.boxes.id is None:
                continue

            keypoints = result.keypoints.xy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()
            boxes_xyxy = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else []

            for index, (kp, track_id) in enumerate(zip(keypoints, track_ids)):
                bbox = self.normalize_bbox_xyxy(boxes_xyxy[index]) if index < len(boxes_xyxy) else None
                if bbox is None:
                    bbox = self.calculate_bbox_from_keypoints(kp)
                if bbox is None:
                    continue

                detections.append({
                    'track_id': track_id,
                    'bbox': bbox,
                    'keypoints': kp,
                    'source': 'pose'
                })

        return detections

    def extract_person_detections(self, results):
        detections = []
        if results is None:
            return detections

        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue

            track_ids = result.boxes.id.int().cpu().tolist()
            boxes_xyxy = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else []
            classes = (
                result.boxes.cls.int().cpu().tolist()
                if result.boxes.cls is not None
                else [0] * len(track_ids)
            )

            for index, track_id in enumerate(track_ids):
                if index < len(classes) and classes[index] != 0:
                    continue

                bbox = self.normalize_bbox_xyxy(boxes_xyxy[index]) if index < len(boxes_xyxy) else None
                if bbox is None:
                    continue

                detections.append({
                    'track_id': track_id,
                    'bbox': bbox,
                    'keypoints': None,
                    'source': 'person'
                })

        return detections

    def match_pose_keypoints_to_person_tracks(self, pose_detections, frame_count):
        assigned_person_ids = set()

        for detection in pose_detections:
            pose_bbox = detection.get('bbox')
            keypoints = detection.get('keypoints')
            pose_center = self.bbox_center(pose_bbox)
            if keypoints is None or pose_bbox is None or pose_center is None:
                continue

            best_person_id = None
            best_rank = None
            for person_id, track in self.person_tracks.items():
                if person_id in assigned_person_ids or track.get('last_seen') != frame_count:
                    continue

                person_bbox = self.latest_track_bbox(track)
                person_center = self.bbox_center(person_bbox)
                if person_bbox is None or person_center is None:
                    continue

                iou = self._bbox_iou(pose_bbox, person_bbox)
                distance = math.hypot(pose_center[0] - person_center[0], pose_center[1] - person_center[1])
                if iou < self.pose_person_match_iou_threshold and distance > self.pose_person_match_center_distance:
                    continue

                rank = (iou, -distance)
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    best_person_id = person_id

            if best_person_id is None:
                continue

            track = self.person_tracks[best_person_id]
            track['keypoints'] = keypoints
            track['keypoints_frame_count'] = frame_count
            track['pose_bbox'] = pose_bbox
            track['pose_missing_frames'] = 0
            assigned_person_ids.add(best_person_id)

    def update_pose_availability(self, frame_count):
        for track in self.person_tracks.values():
            if track.get('last_seen') != frame_count:
                continue
            if track.get('keypoints_frame_count') == frame_count:
                track['pose_missing_frames'] = 0
            else:
                track['pose_missing_frames'] = track.get('pose_missing_frames', 0) + 1

    def has_consistent_person_bbox_track(self, track, frame_count):
        return (
            track.get('last_seen') == frame_count
            and track.get('tracking_source') == 'person'
            and track.get('consecutive_person_bbox_frames', 0)
            >= self.min_consistent_person_bbox_frames
        )

    def update_person_box_tracking(self, detections, frame_count, source_frame=None):
        self.prepare_motion_predictions(frame_count)
        assigned_person_ids = set()

        for detection in detections:
            track_id = detection.get('track_id')
            bbox = detection.get('bbox')
            center = self.bbox_center(bbox)
            if track_id is None or bbox is None or center is None:
                continue

            center_x, center_y = center
            foot_y = float(bbox[1] + bbox[3])
            person_id = self.stable_person_id_for_track(
                track_id,
                (center_x, center_y),
                bbox,
                frame_count,
                assigned_person_ids
            )

            if person_id not in self.person_tracks:
                self.person_tracks[person_id] = self.create_track_state(frame_count)

            track_state = self.person_tracks[person_id]
            previous_tracker_id = track_state.get('last_tracker_id')
            previous_last_seen = track_state.get('last_seen')
            previous_foot_y = track_state.get('foot_y')
            tracker_changed = previous_tracker_id is not None and previous_tracker_id != track_id
            foot_jumped = (
                previous_foot_y is not None
                and abs(float(foot_y) - float(previous_foot_y)) > self.line_crossing_max_jump_px
            )
            if foot_jumped:
                self.line_crossing_people.pop(person_id, None)
                track_state.setdefault(
                    'foot_y_history',
                    deque(maxlen=max(6, self.line_crossing_direction_frames + 2))
                ).clear()
                self.debug_log(
                    (
                        "reset line crossing state after bbox-foot jump "
                        f"foot_jump={abs(float(foot_y) - float(previous_foot_y)) if previous_foot_y is not None else 0:.1f}"
                    ),
                    person_id,
                    force=True
                )
            elif tracker_changed and person_id not in self.pending_line_crossings:
                self.line_crossing_people.pop(person_id, None)
                track_state.setdefault(
                    'foot_y_history',
                    deque(maxlen=max(6, self.line_crossing_direction_frames + 2))
                ).clear()
                self.debug_log(
                    (
                        f"tracker id changed {previous_tracker_id}->{track_id}; "
                        "reset crossing arm while preserving stable person identity"
                    ),
                    person_id,
                    force=True
                )

            assigned_person_ids.add(person_id)
            self.correct_track_motion(track_state, bbox, frame_count)
            if not tracker_changed and previous_last_seen == frame_count - 1:
                track_state['consecutive_detected_frames'] = (
                    track_state.get('consecutive_detected_frames', 0) + 1
                )
            else:
                track_state['consecutive_detected_frames'] = 1
            track_state.setdefault('tracker_ids', set()).add(track_id)
            track_state['last_tracker_id'] = track_id
            detection_source = detection.get('source', 'unknown')
            track_state['tracking_source'] = detection_source
            if detection_source == 'person':
                previous_person_bbox_frame = track_state.get('last_person_bbox_frame')
                if not tracker_changed and previous_person_bbox_frame == frame_count - 1:
                    track_state['consecutive_person_bbox_frames'] = (
                        track_state.get('consecutive_person_bbox_frames', 0) + 1
                    )
                else:
                    track_state['consecutive_person_bbox_frames'] = 1
                track_state['last_person_bbox_frame'] = frame_count
            track_state['positions'].append((center_x, center_y))
            track_state['last_seen'] = frame_count
            keypoints = detection.get('keypoints')
            track_state['keypoints'] = keypoints
            track_state['keypoints_frame_count'] = frame_count if keypoints is not None else None
            track_state['foot_y'] = foot_y
            track_state.setdefault(
                'foot_y_history',
                deque(maxlen=max(6, self.line_crossing_direction_frames + 2))
            ).append(foot_y)
            
            if bbox:
                track_state['bbox_history'].append(bbox)
                if source_frame is not None and bbox[2] > 10 and bbox[3] > 10:
                    snapshot = self.create_person_tracking_snapshot(
                        source_frame,
                        bbox,
                        frame_count,
                        person_id=person_id,
                        tracker_id=track_id,
                        tracking_source=detection_source
                    )
                    if snapshot is not None:
                        track_state['event_snapshots'].append(snapshot)
                        self.remember_pending_post_crossing_snapshot(person_id, snapshot)
        
        current_frame = frame_count
        inactive_ids = [pid for pid, track in self.person_tracks.items() 
                       if current_frame - track['last_seen'] > self.track_max_age_frames
                       and pid not in self.pending_line_crossings]
        for pid in inactive_ids:
            del self.person_tracks[pid]
            self.missed_frisking_people.discard(pid)
            self.frisked_people.discard(pid)
            self.security_people.discard(pid)
            self.line_crossing_people.pop(pid, None)
            self.pending_line_crossings.pop(pid, None)
            self.pending_crossing_grace_frames.pop(pid, None)
            self.pending_crossing_snapshots.pop(pid, None)
            self.security_overlap_frames.pop(pid, None)
            self.person_security_history.pop(pid, None)
            self.track_id_aliases = {
                raw_id: person_id
                for raw_id, person_id in self.track_id_aliases.items()
                if person_id != pid
            }

    def update_person_tracking(self, keypoints_list, track_ids_list, frame_count, source_frame=None):
        detections = []
        for keypoints, track_id in zip(keypoints_list, track_ids_list):
            valid_points = keypoints[(keypoints[:, 0] > 0) | (keypoints[:, 1] > 0)]
            if len(valid_points) == 0:
                continue

            bbox = self.calculate_bbox_from_keypoints(keypoints)
            if bbox is None:
                continue

            detections.append({
                'track_id': track_id,
                'bbox': bbox,
                'keypoints': keypoints,
                'source': 'pose'
            })

        self.update_person_box_tracking(detections, frame_count, source_frame)
    
    def draw_bbox_with_label(self, frame, bbox, label, color=(0, 0, 255), thickness=2):
        x, y, w, h = map(int, bbox)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

        if not label:
            return frame
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        text_thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
        label_top = y - text_height - 10
        label_bottom = y
        text_y = y - 5
        if label_top < 0:
            label_top = y + 2
            label_bottom = y + text_height + baseline + 10
            text_y = y + text_height + 5
        
        cv2.rectangle(frame, (x, label_top),
                     (x + text_width, label_bottom), color, -1)
        
        cv2.putText(frame, label, (x, text_y), font, font_scale,
                   (255, 255, 255), text_thickness)
        
        return frame
    
    def create_clean_frame_for_saving(self, frame, person_id, bbox=None, label=""):
        clean_frame = frame.copy()
        clean_frame = self.draw_horizontal_line(clean_frame)
        
        if bbox is not None:
            clean_frame = self.draw_bbox_with_label(
                clean_frame, bbox, label,
                color=(0, 0, 255), thickness=3
            )
        elif person_id in self.person_tracks:
            track = self.person_tracks[person_id]
            if track['bbox_history']:
                bbox = track['bbox_history'][-1]
                clean_frame = self.draw_bbox_with_label(
                    clean_frame, bbox, label,
                    color=(0, 0, 255), thickness=3
                )
        
        return clean_frame

    def latest_track_bbox(self, track):
        predicted_bbox = self.current_predicted_bbox(track)
        if predicted_bbox is not None:
            return predicted_bbox

        snapshot = track.get('event_snapshots')[-1] if track.get('event_snapshots') else None
        if snapshot and snapshot.get('bbox'):
            return snapshot['bbox']

        bbox_history = track.get('bbox_history')
        if bbox_history:
            return bbox_history[-1]

        return None

    def create_person_tracking_snapshot(
        self,
        frame,
        bbox,
        frame_count,
        person_id=None,
        tracker_id=None,
        tracking_source=None
    ):
        return {
            # The same immutable original frame is shared by every person in this
            # detector frame, avoiding one full-frame copy per tracked person.
            'frame': frame,
            'bbox': bbox,
            'tracking_bbox': bbox,
            'frame_count': frame_count,
            'frame_time': time.time(),
            'person_id': person_id,
            'tracker_id': tracker_id,
            'tracking_source': tracking_source,
            'is_person_crop': False
        }

    def track_snapshot_for_frame(self, track, frame_count):
        for snapshot in reversed(list(track.get('event_snapshots') or [])):
            if snapshot.get('frame_count') == frame_count:
                return snapshot
        return None

    def tracking_evidence_counts(self, evidence):
        if not evidence:
            return 0, 0, 0

        crossing_frame_count = evidence.get('frame_count')
        if crossing_frame_count is None:
            return 0, 0, 0

        snapshots = evidence.get('tracking_snapshots') or []
        pre_count = sum(
            item.get('frame_count', crossing_frame_count) < crossing_frame_count
            for item in snapshots
        )
        crossing_count = sum(
            item.get('frame_count') == crossing_frame_count
            for item in snapshots
        )
        post_count = sum(
            item.get('frame_count', crossing_frame_count) > crossing_frame_count
            for item in snapshots
        )
        return pre_count, crossing_count, post_count

    def snapshot_matches_crossing_identity(self, evidence, snapshot):
        if not evidence or not snapshot:
            return False

        evidence_person_id = evidence.get('person_id')
        snapshot_person_id = snapshot.get('person_id')
        if evidence_person_id is not None and snapshot_person_id != evidence_person_id:
            return False

        if TRACKING_EVENT_REQUIRE_SAME_TRACKER_ID:
            evidence_tracker_id = evidence.get('tracker_id')
            snapshot_tracker_id = snapshot.get('tracker_id')
            if evidence_tracker_id is None or snapshot_tracker_id != evidence_tracker_id:
                return False

        return True

    def has_complete_tracking_evidence(self, evidence):
        if not TRACKING_EVENT_REQUIRE_COMPLETE_WINDOW:
            return True
        if not evidence or not evidence.get('boundary_anchored'):
            return False
        if not all(
            self.snapshot_matches_crossing_identity(evidence, snapshot)
            for snapshot in evidence.get('tracking_snapshots') or []
        ):
            return False

        pre_count, crossing_count, post_count = self.tracking_evidence_counts(evidence)
        return (
            pre_count >= TRACKING_EVENT_PRE_CROSSING_FRAMES
            and crossing_count >= 1
            and post_count >= TRACKING_EVENT_POST_CROSSING_FRAMES
        )

    def tracking_evidence_action(self, evidence, frames_since_crossing, can_collect_more):
        counts = self.tracking_evidence_counts(evidence)
        if self.has_complete_tracking_evidence(evidence):
            return 'complete', counts

        pre_count, crossing_count, post_count = counts
        can_be_completed = (
            bool(evidence and evidence.get('boundary_anchored'))
            and pre_count >= TRACKING_EVENT_PRE_CROSSING_FRAMES
            and crossing_count >= 1
            and post_count < TRACKING_EVENT_POST_CROSSING_FRAMES
        )
        if (
            can_collect_more
            and can_be_completed
            and frames_since_crossing < self.max_line_crossing_grace_frames
        ):
            return 'wait', counts
        return 'skip', counts

    def append_post_crossing_snapshot(self, evidence, snapshot):
        if not evidence or not snapshot or TRACKING_EVENT_POST_CROSSING_FRAMES <= 0:
            return
        if not self.snapshot_matches_crossing_identity(evidence, snapshot):
            return

        crossing_frame_count = evidence.get('frame_count')
        snapshot_frame_count = snapshot.get('frame_count')
        if (
            crossing_frame_count is None
            or snapshot_frame_count is None
            or snapshot_frame_count <= crossing_frame_count
        ):
            return

        tracking_snapshots = evidence.setdefault('tracking_snapshots', [])
        _pre_count, _crossing_count, post_count = self.tracking_evidence_counts(evidence)
        if post_count >= TRACKING_EVENT_POST_CROSSING_FRAMES:
            return
        if any(item.get('frame_count') == snapshot_frame_count for item in tracking_snapshots):
            return

        tracking_snapshots.append(dict(snapshot))

    def remember_line_boundary_crossing_snapshot(self, person_id, crossing_data, track, frame_count):
        snapshot = self.track_snapshot_for_frame(track or {}, frame_count)
        if snapshot is None:
            return

        bbox = snapshot.get('tracking_bbox') or snapshot.get('bbox')
        frame = snapshot.get('frame')
        if frame is None or not bbox or bbox[2] <= 10 or bbox[3] <= 10:
            return

        crossing_person_id = snapshot.get('person_id')
        crossing_tracker_id = snapshot.get('tracker_id')
        if crossing_person_id != person_id:
            return
        if TRACKING_EVENT_REQUIRE_SAME_TRACKER_ID and crossing_tracker_id is None:
            return

        pre_crossing_snapshots = [
            dict(item)
            for item in list(track.get('event_snapshots') or [])
            if item.get('frame_count') is not None
            and item.get('frame_count') < frame_count
            and item.get('person_id') == crossing_person_id
            and (
                not TRACKING_EVENT_REQUIRE_SAME_TRACKER_ID
                or item.get('tracker_id') == crossing_tracker_id
            )
        ][-TRACKING_EVENT_PRE_CROSSING_FRAMES:]
        crossing_tracking_snapshot = dict(snapshot)
        crossing_data['boundary_snapshot'] = {
            'frame': frame,
            'bbox': bbox,
            'frame_count': frame_count,
            'person_id': crossing_person_id,
            'tracker_id': crossing_tracker_id,
            'boundary_anchored': True,
            'tracking_snapshots': pre_crossing_snapshots + [crossing_tracking_snapshot]
        }
        self.debug_log(
            f"line boundary evidence captured pre_crossing={len(pre_crossing_snapshots)} crossing=1",
            person_id,
            force=True
        )

    def remember_boundary_post_crossing_snapshot(self, crossing_data, track, frame_count):
        evidence = crossing_data.get('boundary_snapshot')
        snapshot = self.track_snapshot_for_frame(track or {}, frame_count)
        self.append_post_crossing_snapshot(evidence, snapshot)

    def remember_pending_crossing_snapshot(self, person_id, track, original_frame, frame_count):
        crossing_data = self.line_crossing_people.get(person_id) or {}
        boundary_snapshot = crossing_data.get('boundary_snapshot')
        if boundary_snapshot:
            self.pending_crossing_snapshots[person_id] = {
                **boundary_snapshot,
                'tracking_snapshots': [
                    dict(snapshot)
                    for snapshot in boundary_snapshot.get('tracking_snapshots', [])
                ]
            }
            pre_count, crossing_count, post_count = self.tracking_evidence_counts(
                self.pending_crossing_snapshots[person_id]
            )
            self.debug_log(
                (
                    "tracking evidence anchored at line boundary "
                    f"pre={pre_count} crossing={crossing_count} post={post_count}"
                ),
                person_id,
                force=True
            )
            return

        bbox = self.latest_track_bbox(track)
        if not bbox or bbox[2] <= 10 or bbox[3] <= 10:
            self.debug_log(
                f"line crossing snapshot skipped because bbox is invalid: {bbox}",
                person_id,
                force=True
            )
            return

        crossing_frame = original_frame.copy()
        pre_crossing_snapshots = [
            dict(snapshot)
            for snapshot in list(track.get('event_snapshots') or [])
            if snapshot.get('frame_count') is not None
            and snapshot.get('frame_count') < frame_count
        ][-TRACKING_EVENT_PRE_CROSSING_FRAMES:]
        crossing_tracking_snapshot = self.create_person_tracking_snapshot(
            crossing_frame,
            bbox,
            frame_count,
            person_id=person_id,
            tracker_id=track.get('last_tracker_id'),
            tracking_source=track.get('tracking_source')
        )

        self.pending_crossing_snapshots[person_id] = {
            'frame': crossing_frame,
            'bbox': bbox,
            'frame_count': frame_count,
            'person_id': person_id,
            'tracker_id': track.get('last_tracker_id'),
            'boundary_anchored': False,
            'tracking_snapshots': pre_crossing_snapshots + [crossing_tracking_snapshot]
        }
        self.debug_log(
            f"tracking evidence frozen pre_crossing={len(pre_crossing_snapshots)} crossing=1",
            person_id,
            force=True
        )

    def remember_pending_post_crossing_snapshot(self, person_id, snapshot):
        pending = self.pending_crossing_snapshots.get(person_id)
        self.append_post_crossing_snapshot(pending, snapshot)

    def is_crossing_person_security(self, frame, bbox, security_bboxes=None):
        if not bbox or bbox[2] <= 10 or bbox[3] <= 10:
            return False

        if security_bboxes and self.bbox_overlaps_any_security(bbox, security_bboxes, iou_threshold=0.45):
            return True

        return self.is_security_personnel(frame, bbox)

    def should_skip_missed_for_security(self, person_id, frame, bbox, security_bboxes=None):
        if person_id in self.security_people:
            return True

        if self.security_overlap_frames.get(person_id, 0) < self.security_missed_skip_frames:
            return False

        return self.is_crossing_person_security(frame, bbox, security_bboxes)

    def is_security_personnel(self, frame, bbox):
        x, y, w, h = bbox
        frame_h, frame_w = frame.shape[:2]
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(frame_w, int(x + w))
        y2 = min(frame_h, int(y + h))

        if x2 <= x1 or y2 <= y1:
            return False

        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            return False

        try:
            class_ids, selected_boxes, selected_scores = yolov8_detect_security(person_crop, SECURITY_PERSONNEL_TOL)
            for box, class_id, score in zip(selected_boxes, class_ids, selected_scores):
                if class_id == 0:
                    bx1, by1, bx2, by2 = box
                    box_area = max(0, bx2 - bx1) * max(0, by2 - by1)
                    crop_area = w * h
                    if crop_area > 0 and (box_area / crop_area) > 0.4:
                        return True
        except Exception as e:
            print(f"Error checking security personnel: {e}")

        return False

    def detect_all_security_bboxes(self, frame):
        try:
            class_ids, selected_boxes, selected_scores = yolov8_detect_security(frame, SECURITY_PERSONNEL_TOL)
            security_bboxes = []
            for box, class_id, score in zip(selected_boxes, class_ids, selected_scores):
                if class_id == 0:
                    x1, y1, x2, y2 = box
                    security_bboxes.append([x1, y1, x2 - x1, y2 - y1])
            return security_bboxes
        except Exception as e:
            return []

    def _bbox_iou(self, bbox_a, bbox_b):
        ax1, ay1, aw, ah = bbox_a
        bx1, by1, bw, bh = bbox_b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        if inter_area == 0:
            return 0.0

        area_a = aw * ah
        area_b = bw * bh
        union_area = area_a + area_b - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def bbox_overlaps_any_security(self, person_bbox, security_bboxes, iou_threshold=0.45):
        for sec_bbox in security_bboxes:
            iou = self._bbox_iou(person_bbox, sec_bbox)
            if iou >= iou_threshold:
                return True
        return False

    def bbox_matches_security_detection(self, person_bbox, security_bboxes, iou_threshold=0.20):
        if not person_bbox or not security_bboxes:
            return False

        px, py, pw, ph = person_bbox
        person_center = (px + pw / 2.0, py + ph / 2.0)
        for sec_bbox in security_bboxes:
            if self._bbox_iou(person_bbox, sec_bbox) >= iou_threshold:
                return True

            sx, sy, sw, sh = sec_bbox
            pad_x = sw * 0.15
            pad_y = sh * 0.15
            if (
                sx - pad_x <= person_center[0] <= sx + sw + pad_x
                and sy - pad_y <= person_center[1] <= sy + sh + pad_y
            ):
                return True

        return False

    def security_pose_tracks_for_frame(self, security_bboxes, frame_count, exclude_person_id=None):
        guard_tracks = []
        for guard_id, guard_track in self.person_tracks.items():
            if guard_id == exclude_person_id or guard_track.get('last_seen') != frame_count:
                continue

            guard_kp = self.current_keypoints_for_track(guard_track, frame_count)
            guard_bbox = self.latest_track_bbox(guard_track)
            if guard_kp is None or not guard_bbox:
                continue

            if guard_id in self.security_people or self.bbox_matches_security_detection(guard_bbox, security_bboxes):
                guard_tracks.append((guard_id, guard_track))

        return guard_tracks

    def guard_reach_targets_for_frame(self, security_bboxes, frame_count):
        assignments = {}
        guard_tracks = self.security_pose_tracks_for_frame(security_bboxes, frame_count)
        if not guard_tracks:
            return assignments

        candidates = []
        for person_id, track in self.person_tracks.items():
            if track.get('last_seen') != frame_count or person_id in self.security_people:
                continue

            bbox = self.latest_track_bbox(track)
            if not bbox:
                continue

            keypoints = self.current_keypoints_for_track(track, frame_count)
            if keypoints is None:
                continue

            height = self.keypoint_body_height(keypoints)
            torso = self.mean_valid_keypoints(keypoints, [5, 6, 11, 12])
            if height < 40 or torso is None:
                continue

            candidates.append((person_id, track, keypoints, height, torso))

        for guard_id, guard_track in guard_tracks:
            guard_kp = self.current_keypoints_for_track(guard_track, frame_count)
            if guard_kp is None:
                continue

            guard_height = self.keypoint_body_height(guard_kp)
            guard_torso = self.mean_valid_keypoints(guard_kp, [5, 6, 11, 12])
            if guard_height < 40 or guard_torso is None:
                continue

            best = None
            for person_id, _track, suspect_kp, suspect_height, suspect_torso in candidates:
                if person_id == guard_id:
                    continue

                reach, score = self.is_reaching_towards_person(
                    guard_kp,
                    suspect_kp,
                    guard_height,
                    suspect_height
                )
                method = 'strict'
                if not reach or score < self.frisking_reach_score_threshold:
                    quick_reach, quick_score = self.is_quick_frisking_contact(
                        guard_kp,
                        suspect_kp,
                        guard_height,
                        suspect_height
                    )
                    if not quick_reach or quick_score < self.quick_frisking_contact_score_threshold:
                        continue

                    reach = True
                    score = quick_score
                    method = 'quick'

                torso_distance = float(np.linalg.norm(guard_torso - suspect_torso))
                rank = (score, 1 if method == 'strict' else 0, -torso_distance)
                if best is None or rank > best['rank']:
                    best = {
                        'person_id': person_id,
                        'guard_id': guard_id,
                        'score': score,
                        'distance': torso_distance,
                        'method': method,
                        'rank': rank
                    }

            if best is None:
                continue

            current = assignments.get(best['person_id'])
            if current is None or best['rank'] > current['rank']:
                assignments[best['person_id']] = best

        return assignments

    def is_near_security(self, person_bbox, security_bboxes):
        # A person is near security if their bounding boxes intersect or are very close horizontally
        for sec_bbox in security_bboxes:
            ax1, ay1, aw, ah = person_bbox
            bx1, by1, bw, bh = sec_bbox
            ax2, ay2 = ax1 + aw, ay1 + ah
            bx2, by2 = bx1 + bw, by1 + bh

            inter_x1 = max(ax1, bx1)
            inter_y1 = max(ay1, by1)
            inter_x2 = min(ax2, bx2)
            inter_y2 = min(ay2, by2)

            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            h_dist = max(0, max(ax1, bx1) - min(ax2, bx2))
            
            if inter_area > 0 or h_dist < 40:
                return True
        return False

    def emit_missed_frisking_event(
        self,
        frame,
        original_frame,
        person_id,
        track,
        missed_frisking_events,
        event_frame=None,
        event_bbox=None
    ):
        track = track or {}
        bbox = event_bbox or self.latest_track_bbox(track)
        if not bbox and not track.get('bbox_history'):
            self.debug_log(
                "MISSED decision skipped event because track has no bbox history",
                person_id,
                force=True
            )
            return None

        if not bbox or bbox[2] <= 10 or bbox[3] <= 10:
            self.debug_log(
                f"MISSED decision skipped event because bbox is invalid: {bbox}",
                person_id,
                force=True
            )
            return None

        if event_frame is None:
            snapshot = self.pending_crossing_snapshots.get(person_id)
            if snapshot:
                event_frame = snapshot.get('frame')
                bbox = event_bbox or snapshot.get('bbox') or bbox

        if event_frame is None and track:
            snapshot = track.get('event_snapshots')[-1] if track.get('event_snapshots') else None
            if snapshot and not snapshot.get('is_person_crop'):
                event_frame = snapshot.get('frame')
                bbox = event_bbox or snapshot.get('bbox') or bbox

        if event_frame is None:
            event_frame = original_frame

        self.missed_frisking_people.add(person_id)

        frame = self.draw_bbox_with_label(
            frame, bbox, "MISSED FRISKING",
            color=(0, 0, 255), thickness=3
        )

        clean_frame = self.create_clean_frame_for_saving(
            event_frame.copy(),
            person_id,
            bbox,
            "MISSED FRISKING"
        )

        missed_frisking_events.append({
            "person_id": person_id,
            "bbox": bbox,
            "bbox_xyxy": self.bbox_xywh_to_xyxy(bbox),
            "event_frame": clean_frame
        })

        counts = self.evidence_counts(person_id, self.recent_interaction_window)
        self.debug_log(
            (
                "MISSED event emitted "
                f"recent_near={counts['near_security']} "
                f"recent_near_still={counts['near_still']} "
                f"recent_pose={counts['pose_frisking']} "
                f"recent_score={counts['frisking_score']}/{self.frisking_score_threshold} "
                f"bbox={bbox}"
            ),
            person_id,
            force=True
        )

        return clean_frame

    def resolve_lost_pending_crossings(self, frame, missed_frisking_events, previous_track_events, frame_count):
        missed_frisking_event_frame = None

        for person_id, crossed_at in list(self.pending_line_crossings.items()):
            track = self.person_tracks.get(person_id)
            if track and track.get('last_seen') == frame_count:
                continue

            recent_counts = self.evidence_counts(person_id, self.recent_interaction_window)
            lost_age = frame_count - track.get('last_seen', crossed_at) if track else frame_count - crossed_at
            if self.mark_frisked_if_confirmed(person_id, "after track loss"):
                self.pending_line_crossings.pop(person_id, None)
                self.pending_crossing_grace_frames.pop(person_id, None)
                self.pending_crossing_snapshots.pop(person_id, None)
                continue

            frames_since_crossing = frame_count - crossed_at
            grace_frames = self.pending_grace_for_person(person_id)
            remaining = max(0, grace_frames - frames_since_crossing)
            self.debug_log(
                (
                    f"pending grace remaining={remaining} track_lost_age={lost_age} "
                    f"max_grace={self.max_line_crossing_grace_frames - frames_since_crossing} "
                    f"recent_near={recent_counts['near_security']} "
                    f"recent_near_still={recent_counts['near_still']} "
                    f"recent_pose={recent_counts['pose_frisking']} "
                    f"recent_score={recent_counts['frisking_score']}/{self.frisking_score_threshold}"
                ),
                person_id
            )

            if frames_since_crossing < grace_frames:
                continue

            if self.should_extend_pending_grace(person_id, frames_since_crossing):
                self.debug_log(
                    (
                        "pending grace extended after track loss because recent "
                        f"security/check evidence exists; max_grace_remaining="
                        f"{self.max_line_crossing_grace_frames - frames_since_crossing}"
                    ),
                    person_id,
                    force=True
                )
                continue

            snapshot = self.pending_crossing_snapshots.get(person_id)
            evidence_action, evidence_counts = self.tracking_evidence_action(
                snapshot, frames_since_crossing, can_collect_more=False
            )
            if evidence_action != 'complete':
                self.pending_line_crossings.pop(person_id, None)
                self.pending_crossing_grace_frames.pop(person_id, None)
                self.pending_crossing_snapshots.pop(person_id, None)
                self.debug_log(
                    (
                        "missed event skipped after track loss: incomplete crossing evidence "
                        f"pre={evidence_counts[0]}/{TRACKING_EVENT_PRE_CROSSING_FRAMES} "
                        f"crossing={evidence_counts[1]}/1 "
                        f"post={evidence_counts[2]}/{TRACKING_EVENT_POST_CROSSING_FRAMES}"
                    ),
                    person_id,
                    force=True
                )
                continue

            self.pending_line_crossings.pop(person_id, None)
            self.pending_crossing_grace_frames.pop(person_id, None)
            snapshot = self.pending_crossing_snapshots.pop(person_id, None)
            if person_id in self.missed_frisking_people:
                continue

            event_frame = snapshot.get('frame') if snapshot else None
            event_bbox = snapshot.get('bbox') if snapshot else self.latest_track_bbox(track or {})
            security_frame = event_frame if event_frame is not None else frame
            security_bboxes = self.detect_all_security_bboxes(security_frame) if security_frame is not None else []
            if self.should_skip_missed_for_security(person_id, security_frame, event_bbox, security_bboxes):
                self.mark_security_person(
                    person_id,
                    "crossing person is SECURITY; skipped missed event after track loss"
                )
                continue

            self.debug_log(
                (
                    "MISSED decision after track loss "
                    f"recent_pose={recent_counts['pose_frisking']}/{self.strong_pose_latch_frames} "
                    f"recent_near_still={recent_counts['near_still']}/{self.min_near_still_frames} "
                    f"recent_score={recent_counts['frisking_score']}/{self.frisking_score_threshold} "
                    f"max_reach={recent_counts['max_reach_score']}/{self.frisking_strong_reach_score_threshold} "
                    f"track_lost_age={lost_age}"
                ),
                person_id,
                force=True
            )
            clean_frame = self.emit_missed_frisking_event(
                frame,
                event_frame if event_frame is not None else frame,
                person_id,
                track or {},
                missed_frisking_events,
                event_frame=event_frame,
                event_bbox=event_bbox
            )
            if clean_frame is not None:
                missed_frisking_event_frame = clean_frame.copy()
                previous_track_events.extend(
                    self.build_previous_track_events(
                        person_id,
                        track or {},
                        frame_count,
                        snapshots=snapshot.get('tracking_snapshots') if snapshot else None
                    )
                )

        return missed_frisking_event_frame

    def process_single_frame(self, frame):
        self.frame_count += 1
        frame_count = self.frame_count
        missed_frisking_events = []
        previous_track_events = []
        missed_frisking_event_frame = None
        
        original_frame = frame.copy()
        full_frame_security_bboxes = self.detect_all_security_bboxes(original_frame)
        
        pose_detections = []
        try:
            pose_detections = self.extract_pose_detections(self.process_frame(frame))
        except Exception as exc:
            self.debug_log(
                f"pose inference unavailable; continuing with person bbox tracking: {exc}",
                force=True
            )

        if self.person_model is not None:
            person_detections = []
            try:
                person_detections = self.extract_person_detections(self.process_person_frame(frame))
            except Exception as exc:
                self.debug_log(f"person bbox inference unavailable: {exc}", force=True)
            if person_detections:
                self.update_person_box_tracking(person_detections, frame_count, original_frame)
                self.match_pose_keypoints_to_person_tracks(pose_detections, frame_count)
            else:
                self.debug_log("person model produced no tracks; falling back to pose tracking")
                self.update_person_box_tracking(pose_detections, frame_count, original_frame)
        else:
            self.update_person_box_tracking(pose_detections, frame_count, original_frame)
        self.update_pose_availability(frame_count)
        
        frame = self.draw_horizontal_line(frame)
        guard_reach_targets = self.guard_reach_targets_for_frame(full_frame_security_bboxes, frame_count)
        
        # ----- TEST VISUALIZATION -----
        # Draw all security guards in green
        for sec_bbox in full_frame_security_bboxes:
            frame = self.draw_bbox_with_label(frame, sec_bbox, "SECURITY", color=(0, 255, 0), thickness=2)
            
        # Draw all tracked people
        for pid, track in self.person_tracks.items():
            if self.track_visible_this_frame(track, frame_count):
                p_bbox = self.latest_track_bbox(track)
                if p_bbox:
                    if track.get('motion_prediction_only'):
                        if pid not in self.security_people:
                            age = track.get('motion_prediction_age', 0)
                            frame = self.draw_bbox_with_label(
                                frame, p_bbox, f"{pid} PRED:{age}",
                                color=(0, 165, 255), thickness=2
                            )
                        continue
                    if (
                        pid in self.security_people
                        or self.bbox_matches_security_detection(p_bbox, full_frame_security_bboxes)
                    ):
                        continue # Already drawn as security (green)
                    elif pid in self.frisked_people:
                        frame = self.draw_bbox_with_label(frame, p_bbox, f"{pid} FRISKED", color=(255, 0, 0), thickness=2)
                    elif pid in self.missed_frisking_people:
                        frame = self.draw_bbox_with_label(frame, p_bbox, f"{pid} MISSED", color=(0, 0, 255), thickness=2)
                    elif pid in self.pending_line_crossings:
                        elapsed = frame_count - self.pending_line_crossings[pid]
                        remaining = max(0, self.pending_grace_for_person(pid) - elapsed)
                        hist = list(self.person_security_history.get(pid, []))
                        recent_hist = hist[-self.recent_interaction_window:]
                        near_still = sum(1 for h in recent_hist if h.get('near_security') and h.get('standing_still'))
                        pose_cnt = sum(1 for h in recent_hist if h.get('pose_frisking'))
                        score_cnt = sum(int(h.get('frisking_score', 0) or 0) for h in recent_hist)
                        label = f"{pid} WAIT:{remaining} S:{near_still} P:{pose_cnt} F:{score_cnt}"
                        frame = self.draw_bbox_with_label(frame, p_bbox, label, color=(0, 165, 255), thickness=2)
                    else:
                        # Show live evidence counters
                        hist = list(self.person_security_history.get(pid, []))
                        recent_hist = hist[-self.recent_interaction_window:]
                        near_still = sum(1 for h in recent_hist if h.get('near_security') and h.get('standing_still'))
                        pose_cnt = sum(1 for h in recent_hist if h.get('pose_frisking'))
                        score_cnt = sum(int(h.get('frisking_score', 0) or 0) for h in recent_hist)
                        if near_still > 0 or pose_cnt > 0 or score_cnt > 0:
                            label = f"{pid} S:{near_still} P:{pose_cnt} F:{score_cnt}"
                            frame = self.draw_bbox_with_label(frame, p_bbox, label, color=(0, 255, 255), thickness=2)
                        else:
                            frame = self.draw_bbox_with_label(frame, p_bbox, pid, color=(255, 255, 0), thickness=2)
        # ------------------------------
        
        for person_id, track in self.person_tracks.items():
            if track.get('last_seen') != frame_count:
                continue

            bbox = self.latest_track_bbox(track)
            
            # 1. Identify if this track is the security guard. Require repeated
            # evidence so a passing person is not promoted to security by one frame.
            if (
                person_id not in self.security_people
                and person_id not in self.missed_frisking_people
                and person_id not in self.frisked_people
            ):
                security_candidate = False
                if bbox and full_frame_security_bboxes and self.bbox_overlaps_any_security(bbox, full_frame_security_bboxes, iou_threshold=0.45):
                    security_candidate = True
                elif bbox and bbox[2] > 10 and bbox[3] > 10 and self.is_security_personnel(original_frame, bbox):
                    security_candidate = True

                if security_candidate:
                    self.security_overlap_frames[person_id] += 1
                    self.debug_log(
                        (
                            "security candidate "
                            f"frames={self.security_overlap_frames[person_id]}/{self.security_confirm_frames}"
                        ),
                        person_id
                    )
                    if self.security_overlap_frames[person_id] >= self.security_confirm_frames:
                        self.mark_security_person(
                            person_id,
                            "classified as SECURITY after repeated confirmation"
                        )
                else:
                    self.security_overlap_frames[person_id] = 0

            # 2. Collect per-frame evidence (do NOT make frisking decision here)
            near_sec = False
            standing_still = False
            pose_frisking = False
            reach_score = 0
            
            if bbox:
                reach_target = guard_reach_targets.get(person_id)
                near_sec = bool(full_frame_security_bboxes and self.is_near_security(bbox, full_frame_security_bboxes))
                if reach_target:
                    near_sec = True
                    pose_frisking = True
                    reach_score = int(reach_target['score'])
                    self.debug_log(
                        (
                            f"guard hand activity from {reach_target['guard_id']} "
                            f"method={reach_target.get('method', 'strict')} "
                            f"reach_score={reach_score} best_target_distance={reach_target['distance']:.1f}"
                        ),
                        person_id
                    )
                
                if near_sec:
                    positions = track.get('positions', [])
                    if len(positions) >= 4:
                        recent = list(positions)[-4:]
                        dist_moved = math.sqrt((recent[-1][0] - recent[0][0])**2 + (recent[-1][1] - recent[0][1])**2)
                        if dist_moved < 35:
                            standing_still = True

            frisking_score = self.frame_frisking_score(
                near_sec,
                standing_still,
                pose_frisking,
                reach_score
            )
            
            # Store this frame's evidence
            self.person_security_history[person_id].append({
                'near_security': near_sec,
                'standing_still': standing_still,
                'pose_frisking': pose_frisking,
                'reach_score': reach_score,
                'frisking_score': frisking_score
            })

            counts = self.evidence_counts(person_id)
            recent_counts = self.evidence_counts(person_id, self.recent_interaction_window)
            self.debug_log(
                (
                    "evidence "
                    f"near={counts['near_security']} "
                    f"near_still={counts['near_still']} "
                    f"pose={counts['pose_frisking']} "
                    f"recent_near={recent_counts['near_security']} "
                    f"recent_near_still={recent_counts['near_still']} "
                    f"recent_pose={recent_counts['pose_frisking']} "
                    f"recent_score={recent_counts['frisking_score']}/{self.frisking_score_threshold} "
                    f"current near={near_sec} still={standing_still} "
                    f"pose={pose_frisking} reach={reach_score} frame_score={frisking_score}"
                ),
                person_id
            )
            if person_id not in self.security_people:
                self.mark_frisked_if_confirmed(person_id, "from evidence")
            
            # 3. Check line crossing using bbox bottom, not body center.
            foot_y = track.get('foot_y')
            if foot_y is not None:
                current_y = foot_y
                
                if self.check_line_crossing(person_id, current_y, bbox):
                    if (
                        self.has_consistent_person_bbox_track(track, frame_count)
                        and track.get('keypoints_frame_count') != frame_count
                    ):
                        self.debug_log(
                            (
                                "crossing accepted from consistent person bbox while pose track is unavailable "
                                f"bbox_frames={track.get('consecutive_person_bbox_frames', 0)} "
                                f"pose_missing_frames={track.get('pose_missing_frames', 0)}"
                            ),
                            person_id,
                            force=True
                        )
                    # Do not decide on the crossing frame. The out camera line can cut
                    # through an active search, so keep collecting guard interaction.
                    recent_counts = self.evidence_counts(person_id, self.recent_interaction_window)
                    self.debug_log(
                        (
                            f"line crossed bbox_bottom_y={current_y:.1f} "
                            f"line_y={self.horizontal_line_y}; "
                            f"recent_near={recent_counts['near_security']} "
                            f"recent_near_still={recent_counts['near_still']} "
                            f"recent_pose={recent_counts['pose_frisking']} "
                            f"recent_score={recent_counts['frisking_score']}/{self.frisking_score_threshold}"
                        ),
                        person_id,
                        force=True
                    )
                    if self.mark_frisked_if_confirmed(person_id, "at crossing"):
                        self.pending_crossing_snapshots.pop(person_id, None)
                        self.debug_log(
                            "line crossed but already FRISKED",
                            person_id,
                            force=True
                        )
                    else:
                        self.pending_line_crossings[person_id] = frame_count
                        grace_frames = self.crossing_grace_for_track(person_id, track)
                        self.pending_crossing_grace_frames[person_id] = grace_frames
                        self.remember_pending_crossing_snapshot(person_id, track, original_frame, frame_count)
                        self.debug_log(
                            (
                                f"starting grace={grace_frames} "
                                f"smooth_frames={track.get('consecutive_detected_frames', 0)} "
                                f"no_security_contact={self.has_no_security_contact(person_id)}"
                            ),
                            person_id,
                            force=True
                        )

            if person_id in self.pending_line_crossings:
                if self.mark_frisked_if_confirmed(person_id, "during grace"):
                    del self.pending_line_crossings[person_id]
                    self.pending_crossing_grace_frames.pop(person_id, None)
                    self.pending_crossing_snapshots.pop(person_id, None)
                    continue

                crossed_at = self.pending_line_crossings[person_id]
                frames_since_crossing = frame_count - crossed_at
                grace_frames = self.pending_grace_for_person(person_id)
                remaining = max(0, grace_frames - frames_since_crossing)
                recent_counts = self.evidence_counts(person_id, self.recent_interaction_window)
                self.debug_log(
                    (
                        f"pending grace remaining={remaining} "
                        f"max_grace={self.max_line_crossing_grace_frames - frames_since_crossing} "
                        f"recent_near={recent_counts['near_security']} "
                        f"recent_near_still={recent_counts['near_still']} "
                        f"recent_pose={recent_counts['pose_frisking']} "
                        f"recent_score={recent_counts['frisking_score']}/{self.frisking_score_threshold}"
                    ),
                    person_id
                )
                if frames_since_crossing >= grace_frames:
                    if self.should_extend_pending_grace(person_id, frames_since_crossing):
                        self.debug_log(
                            (
                                "pending grace extended because person still has recent "
                                f"security/check evidence; max_grace_remaining="
                                f"{self.max_line_crossing_grace_frames - frames_since_crossing}"
                            ),
                            person_id,
                            force=True
                        )
                        continue

                    crossing_snapshot = self.pending_crossing_snapshots.get(person_id)
                    evidence_action, evidence_counts = self.tracking_evidence_action(
                        crossing_snapshot, frames_since_crossing, can_collect_more=True
                    )
                    if evidence_action == 'wait':
                        self.debug_log(
                            (
                                "waiting for complete crossing evidence "
                                f"pre={evidence_counts[0]}/{TRACKING_EVENT_PRE_CROSSING_FRAMES} "
                                f"crossing={evidence_counts[1]}/1 "
                                f"post={evidence_counts[2]}/{TRACKING_EVENT_POST_CROSSING_FRAMES}"
                            ),
                            person_id
                        )
                        continue
                    if evidence_action == 'skip':
                        del self.pending_line_crossings[person_id]
                        self.pending_crossing_grace_frames.pop(person_id, None)
                        self.pending_crossing_snapshots.pop(person_id, None)
                        self.debug_log(
                            (
                                "missed event skipped: incomplete crossing evidence "
                                f"pre={evidence_counts[0]}/{TRACKING_EVENT_PRE_CROSSING_FRAMES} "
                                f"crossing={evidence_counts[1]}/1 "
                                f"post={evidence_counts[2]}/{TRACKING_EVENT_POST_CROSSING_FRAMES}"
                            ),
                            person_id,
                            force=True
                        )
                        continue

                    del self.pending_line_crossings[person_id]
                    self.pending_crossing_grace_frames.pop(person_id, None)
                    crossing_snapshot = self.pending_crossing_snapshots.pop(person_id, None)
                    if person_id not in self.missed_frisking_people:
                        decision_bbox = self.latest_track_bbox(track)
                        if self.should_skip_missed_for_security(person_id, original_frame, decision_bbox, full_frame_security_bboxes):
                            self.mark_security_person(
                                person_id,
                                "crossing person is SECURITY; skipped missed event"
                            )
                            continue

                        self.debug_log(
                            (
                                "MISSED decision after grace "
                                f"recent_pose={recent_counts['pose_frisking']}/{self.strong_pose_latch_frames} "
                                f"recent_near_still={recent_counts['near_still']}/{self.min_near_still_frames} "
                                f"recent_score={recent_counts['frisking_score']}/{self.frisking_score_threshold} "
                                f"max_reach={recent_counts['max_reach_score']}/{self.frisking_strong_reach_score_threshold}"
                            ),
                            person_id,
                            force=True
                        )
                        clean_frame = self.emit_missed_frisking_event(
                            frame,
                            original_frame,
                            person_id,
                            track,
                            missed_frisking_events,
                            event_frame=(
                                crossing_snapshot.get('frame')
                                if crossing_snapshot else original_frame
                            ),
                            event_bbox=(
                                crossing_snapshot.get('bbox')
                                if crossing_snapshot else decision_bbox
                            )
                        )
                        if clean_frame is not None:
                            missed_frisking_event_frame = clean_frame.copy()
                            previous_track_events.extend(
                                self.build_previous_track_events(
                                    person_id,
                                    track,
                                    frame_count,
                                    snapshots=(
                                        crossing_snapshot.get('tracking_snapshots')
                                        if crossing_snapshot else None
                                    )
                                )
                            )
                            
        lost_pending_frame = self.resolve_lost_pending_crossings(
            frame,
            missed_frisking_events,
            previous_track_events,
            frame_count
        )
        if lost_pending_frame is not None:
            missed_frisking_event_frame = lost_pending_frame

        return missed_frisking_event_frame if missed_frisking_event_frame is not None else frame, missed_frisking_events, previous_track_events

def open_rtsp_stream(rtsp_url, retries=5, delay=3):
    """Open an RTSP stream with retry logic."""
    for attempt in range(1, retries + 1):
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            print(f"RTSP stream opened successfully (attempt {attempt}).")
            return cap
        print(f"Attempt {attempt}/{retries}: Could not open RTSP stream. Retrying in {delay}s...")
        cap.release()
        time.sleep(delay)
    return None


def main():
    upload_config = load_upload_config()

    # ── Hardcoded RTSP configuration ─────────────────────────────────
    rtsp_url = "rtsp://admin:Admin@123@192.168.121.9:554/video/live?channel=1&subtype=0"
    model_path = "yolo11l-pose.pt"
    reconnect_delay = 5
    # ─────────────────────────────────────────────────────────────────

    print(f"RTSP URL: {rtsp_url}")

    cap = open_rtsp_stream(rtsp_url)
    if cap is None:
        print(f"Error: Could not open RTSP stream at {rtsp_url} after multiple attempts.")
        return

    print("Initializing Frisking Detector...")
    detector = FriskingDetector(model_path=model_path)
    
    print("Starting RTSP stream processing... Press 'q' to quit.")
    
    cv2.namedWindow('Frisking Missed - RTSP Out', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Frisking Missed - RTSP Out', 1280, 720)

    consecutive_failures = 0
    max_consecutive_failures = 30
    previous_tracking_frames = deque(maxlen=TRACKING_HISTORY_MAX_FRAMES)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                print(f"Lost RTSP stream after {consecutive_failures} consecutive failures. Reconnecting...")
                cap.release()
                time.sleep(reconnect_delay)
                cap = open_rtsp_stream(rtsp_url)
                if cap is None:
                    print("Failed to reconnect. Exiting.")
                    break
                consecutive_failures = 0
            continue

        consecutive_failures = 0
            
        processed_frame, missed_events, _ = detector.process_single_frame(frame)

        tracking_frame_record = make_tracking_frame_record(
            processed_frame,
            detector.frame_count,
            upload_config.get('label', 'frisking_missed'),
        )
        prune_tracking_history(previous_tracking_frames, tracking_frame_record["frame_time"])

        for event in missed_events:
            bbox_xyxy = event.get('bbox_xyxy')
            person_id = event.get('person_id', 'unknown')
            event_frame = event.get('event_frame')
            if bbox_xyxy and event_frame is not None:
                print(f"EVENT DETECTED: {person_id} | bbox_xyxy={bbox_xyxy}")
                label = upload_config.get('label', 'frisking_missed')
                tracking_images = build_tracking_images(
                    sample_tracking_records(previous_tracking_frames, tracking_frame_record),
                    bbox_xyxy,
                    label,
                )
                upload_event_to_server(upload_config, event_frame, bbox_xyxy, person_id, tracking_images=tracking_images)
        previous_tracking_frames.append(tracking_frame_record)
        
        # processed_frame already has the frisking line and bounding boxes drawn by the detector
        cv2.imshow('Frisking Missed - RTSP Out', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
