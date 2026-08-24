"""YOLO person tracking, player registry, and per-player motion detection."""

import time
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import (
    CONFIRM_SECONDS,
    TRACK_LOST_TIMEOUT,
    PERSON_MODEL,
    USE_REID_TRACKER,
    PERSON_CLASS_ID,
    DETECTION_CONFIDENCE,
    PERSON_INFERENCE_SIZE,
    PERSON_TRACK_INTERVAL_FRAMES,
    PERSON_ASYNC_TRACKING,
    PERSON_ASYNC_MIN_INTERVAL_SEC,
    MOTION_AREA_RATIO_THRESHOLD,
    BLUR_KERNEL,
    MOTION_PIXEL_DIFF_THRESHOLD,
    MOTION_MEAN_DIFF_THRESHOLD,
    MOTION_CENTER_SHIFT_THRESHOLD,
    MOTION_SIZE_CHANGE_THRESHOLD,
    MOTION_ROI_INSET_X,
    MOTION_ROI_INSET_Y,
    MOTION_REFERENCE_BLEND,
)


@dataclass
class PlayerRecord:
    track_id: int
    first_seen: float
    last_seen: float
    bbox: tuple = field(default=None)
    name: str = "Unknown"

    def age(self, now: float) -> float:
        return now - self.first_seen

    def is_confirmed(self, now: float) -> bool:
        return self.age(now) >= CONFIRM_SECONDS

    def is_stale(self, now: float) -> bool:
        return (now - self.last_seen) > TRACK_LOST_TIMEOUT


class PlayerRegistry:
    def __init__(self):
        self.players: dict[int, PlayerRecord] = {}
        self.locked = False

    def lock_players(self):
        """بعد بداية اللعبة: لا تسمح بإضافة أي ID جديد."""
        self.locked = True
        print("[PLAYERS] Player list locked. New players will be ignored.")

    def update(self, track_ids, boxes):
        now = time.time()

        for tid, box in zip(track_ids, boxes):

            # بعد بداية اللعبة تجاهل أي ID جديد
            if tid not in self.players:
                if self.locked:
                    continue

                self.players[tid] = PlayerRecord(
                    track_id=tid,
                    first_seen=now,
                    last_seen=now,
                    bbox=box
                )

            else:
                self.players[tid].last_seen = now
                self.players[tid].bbox = box

        # قبل اللعبة فقط نحذف الـ IDs القديمة
        if not self.locked:
            stale_ids = [
                tid for tid, rec in self.players.items()
                if rec.is_stale(now)
            ]

            for tid in stale_ids:
                del self.players[tid]

        return now

    def confirmed_players(self, now: float):
        return [p for p in self.players.values() if p.is_confirmed(now)]

    def confirmed_count(self, now: float) -> int:
        return len(self.confirmed_players(now))


class PersonTracker:
    """YOLOv8 + ByteTrack running on a background worker to keep the UI real-time."""

    def __init__(self):
        from ultralytics import YOLO

        self.model = YOLO(PERSON_MODEL)
        self.tracker_cfg = "botsort.yaml" if USE_REID_TRACKER else "bytetrack.yaml"
        self._frame_no = 0
        self._last_track_ids = []
        self._last_boxes = []
        self._lock = threading.Lock()
        self._pending_frame = None
        self._running = True
        self._last_inference_at = 0.0
        self._last_submit_at = 0.0
        self._worker = None

        if PERSON_ASYNC_TRACKING:
            self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="person-tracker")
            self._worker.start()

    def _infer(self, frame_bgr):
        results = self.model.track(
            frame_bgr,
            persist=True,
            classes=[PERSON_CLASS_ID],
            conf=DETECTION_CONFIDENCE,
            tracker=self.tracker_cfg,
            imgsz=PERSON_INFERENCE_SIZE,
            verbose=False,
        )

        track_ids, boxes = [], []
        result = results[0]
        if result.boxes is not None and result.boxes.id is not None:
            ids = result.boxes.id.int().tolist()
            xyxy = result.boxes.xyxy.tolist()
            for tid, box in zip(ids, xyxy):
                track_ids.append(tid)
                boxes.append(tuple(map(int, box)))
        return track_ids, boxes

    def _worker_loop(self):
        while self._running:
            with self._lock:
                frame = self._pending_frame
                self._pending_frame = None

            if frame is None:
                time.sleep(0.002)
                continue

            delay = PERSON_ASYNC_MIN_INTERVAL_SEC - (time.time() - self._last_inference_at)
            if delay > 0:
                time.sleep(delay)

            try:
                track_ids, boxes = self._infer(frame)
                with self._lock:
                    self._last_track_ids = track_ids
                    self._last_boxes = boxes
                self._last_inference_at = time.time()
            except Exception as exc:
                print(f"[WARN] Person tracker inference failed: {exc}")
                time.sleep(0.02)

    def process(self, frame_bgr):
        """Submit the newest frame and immediately return the latest completed tracking result."""
        self._frame_no += 1

        if PERSON_ASYNC_TRACKING:
            # Cap submissions near camera rate and keep only the newest pending frame.
            now = time.time()
            with self._lock:
                if now - self._last_submit_at >= 0.025:
                    self._pending_frame = frame_bgr.copy()
                    self._last_submit_at = now
                return list(self._last_track_ids), list(self._last_boxes)

        if (
            self._last_track_ids
            and PERSON_TRACK_INTERVAL_FRAMES > 1
            and self._frame_no % PERSON_TRACK_INTERVAL_FRAMES != 1
        ):
            return list(self._last_track_ids), list(self._last_boxes)

        track_ids, boxes = self._infer(frame_bgr)
        self._last_track_ids = track_ids
        self._last_boxes = boxes
        return track_ids, boxes

    def close(self):
        self._running = False
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=0.5)


def draw_registry(frame, registry: PlayerRegistry, now: float, eliminated_ids: set = None):
    eliminated_ids = eliminated_ids or set()
    for rec in registry.players.values():
        if rec.bbox is None:
            continue
        x1, y1, x2, y2 = rec.bbox
        confirmed = rec.is_confirmed(now)
        display_name = rec.name if rec.name != "Unknown" else f"ID {rec.track_id}"

        if rec.track_id in eliminated_ids:
            color = (120, 120, 120)
            status = f"{display_name} - OUT"
        elif confirmed:
            color = (0, 200, 0)
            status = f"{display_name} - CONFIRMED"
        else:
            color = (0, 0, 255)
            status = f"{display_name} - NEW ({rec.age(now):.1f}s)"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            status,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return frame


class PerPlayerMotionChecker:
    """Detect deliberate player motion while filtering camera noise and tracker jitter."""

    PATCH_SIZE = (96, 128)

    def __init__(self, threshold: float = MOTION_AREA_RATIO_THRESHOLD):
        self.threshold = threshold
        self.reference_patches = {}
        self.reference_boxes = {}
        self.last_thresh = None
        self.last_scores = {}

    def reset(self):
        """Start a fresh RED LIGHT baseline."""
        self.reference_patches.clear()
        self.reference_boxes.clear()
        self.last_thresh = None
        self.last_scores.clear()

    @staticmethod
    def _normalize_patch(patch):
        patch = cv2.resize(patch, PerPlayerMotionChecker.PATCH_SIZE, interpolation=cv2.INTER_AREA)
        patch = cv2.GaussianBlur(patch, (7, 7), 0)
        patch_f = patch.astype(np.float32)
        patch_f -= float(patch_f.mean())
        return np.clip(patch_f + 128.0, 0, 255).astype(np.uint8)

    @staticmethod
    def _box_metrics(box):
        x1, y1, x2, y2 = map(float, box)
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5, width, height)

    def _extract_patch(self, gray, box):
        h, w = gray.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        ix = int(bw * MOTION_ROI_INSET_X)
        iy = int(bh * MOTION_ROI_INSET_Y)
        rx1 = max(0, x1 + ix)
        ry1 = max(0, y1 + iy)
        rx2 = min(w, x2 - ix)
        ry2 = min(h, y2 - iy)
        if rx2 <= rx1 or ry2 <= ry1:
            return None, None
        crop = gray[ry1:ry2, rx1:rx2]
        if crop.size == 0:
            return None, None
        return self._normalize_patch(crop), (rx1, ry1, rx2, ry2)

    def check(self, frame_bgr, player_boxes: dict) -> set:
        """Return track IDs with sustained, meaningful motion during RED LIGHT."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        moving_ids = set()
        self.last_thresh = np.zeros_like(gray, dtype=np.uint8)
        active_ids = set(player_boxes)

        # Drop baselines for tracks that disappeared.
        for tid in list(self.reference_patches):
            if tid not in active_ids:
                self.reference_patches.pop(tid, None)
                self.reference_boxes.pop(tid, None)
                self.last_scores.pop(tid, None)

        for tid, box in player_boxes.items():
            patch, roi = self._extract_patch(gray, box)
            if patch is None:
                continue

            if tid not in self.reference_patches:
                # The first stable frame after the RED grace period becomes the baseline.
                self.reference_patches[tid] = patch
                self.reference_boxes[tid] = tuple(box)
                self.last_scores[tid] = {"pixel_ratio": 0.0, "mean_diff": 0.0, "center_shift": 0.0, "size_change": 0.0}
                continue

            reference = self.reference_patches[tid]
            diff = cv2.absdiff(reference, patch)
            _, mask = cv2.threshold(diff, MOTION_PIXEL_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            pixel_ratio = float(np.count_nonzero(mask)) / float(mask.size)
            mean_diff = float(diff.mean())

            rcx, rcy, rw, rh = self._box_metrics(self.reference_boxes[tid])
            ccx, ccy, cw, ch = self._box_metrics(box)
            center_shift = float(np.hypot(ccx - rcx, ccy - rcy) / max(rw, rh))
            size_change = max(abs(cw / rw - 1.0), abs(ch / rh - 1.0))

            visual_motion = pixel_ratio >= self.threshold and mean_diff >= MOTION_MEAN_DIFF_THRESHOLD
            geometric_motion = (
                center_shift >= MOTION_CENTER_SHIFT_THRESHOLD
                or size_change >= MOTION_SIZE_CHANGE_THRESHOLD
            )

            self.last_scores[tid] = {
                "pixel_ratio": pixel_ratio,
                "mean_diff": mean_diff,
                "center_shift": center_shift,
                "size_change": size_change,
            }

            if visual_motion or geometric_motion:
                moving_ids.add(tid)
            else:
                # Slowly adapt only while the player is clearly still. This absorbs exposure noise.
                self.reference_patches[tid] = cv2.addWeighted(
                    reference, 1.0 - MOTION_REFERENCE_BLEND, patch, MOTION_REFERENCE_BLEND, 0
                )
                # Small tracker jitter should not permanently move the geometric baseline.
                old_box = np.asarray(self.reference_boxes[tid], dtype=np.float32)
                new_box = np.asarray(box, dtype=np.float32)
                blended = old_box * 0.98 + new_box * 0.02
                self.reference_boxes[tid] = tuple(float(v) for v in blended)

            # Keep an evidence mask in full-frame coordinates.
            rx1, ry1, rx2, ry2 = roi
            mask_full = cv2.resize(mask, (rx2 - rx1, ry2 - ry1), interpolation=cv2.INTER_NEAREST)
            self.last_thresh[ry1:ry2, rx1:rx2] = np.maximum(
                self.last_thresh[ry1:ry2, rx1:rx2], mask_full
            )

        return moving_ids


def cmd_registry_test(use_webcam: bool):
    """Run a standalone player tracking test."""
    from djitellopy import Tello

    if use_webcam:
        from .camera_utils import open_camera
        cap = open_camera()
        get_frame = lambda: cap.read()[1]
    else:
        tello = Tello()
        tello.connect()
        print(f"[INFO] Drone battery: {tello.get_battery()}%")
        tello.streamon()
        get_frame = lambda: cv2.cvtColor(tello.get_frame_read().frame, cv2.COLOR_RGB2BGR)

    tracker = PersonTracker()
    registry = PlayerRegistry()

    print("[INFO] Press 'q' to exit.")
    while True:
        frame = get_frame()
        if frame is None:
            continue

        track_ids, boxes = tracker.process(frame)
        now = registry.update(track_ids, boxes)
        frame = draw_registry(frame, registry, now)

        cv2.putText(
            frame,
            f"confirmed: {registry.confirmed_count(now)} / visible: {len(registry.players)}",
            (10, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
        cv2.imshow("Player Registry Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    if use_webcam:
        cap.release()
    else:
        tello.streamoff()
