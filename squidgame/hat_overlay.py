"""AR hat overlay helpers for tracked players."""

import os
import random

import cv2
import numpy as np

from .config import HATS_DIR, HAT_WIDTH_RATIO, HAT_VERTICAL_OVERLAP_RATIO

_RESIZED_HAT_CACHE = {}


def load_hat_images(hats_dir: str = HATS_DIR) -> dict:
    """Load transparent PNG hats once and return them as BGRA images."""
    hats = {}
    if not os.path.isdir(hats_dir):
        print(f"[WARN] Hat directory not found: {hats_dir}. Hat overlay disabled.")
        return hats

    for fname in sorted(os.listdir(hats_dir)):
        if not fname.lower().endswith(".png"):
            continue
        path = os.path.join(hats_dir, fname)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] Could not read hat image: {path}")
            continue
        if img.ndim != 3 or img.shape[2] != 4:
            print(f"[WARN] Ignored '{fname}'. Hat images must be transparent RGBA PNG files. Shape={img.shape}")
            continue
        hats[fname] = img

    if hats:
        print(f"[INFO] Loaded {len(hats)} hat(s): {', '.join(hats.keys())}")
    else:
        print(f"[WARN] No valid hat images found in {hats_dir}. Hat overlay disabled.")
    return hats


def _overlay_rgba(frame_bgr: np.ndarray, overlay_bgra: np.ndarray, x: int, y: int) -> np.ndarray:
    """Alpha-blend an RGBA image onto a BGR frame with safe edge clipping."""
    h, w = overlay_bgra.shape[:2]
    frame_h, frame_w = frame_bgr.shape[:2]

    x1, y1 = x, y
    x2, y2 = x + w, y + h
    src_x1, src_y1 = 0, 0

    if x1 < 0:
        src_x1 = -x1
        x1 = 0
    if y1 < 0:
        src_y1 = -y1
        y1 = 0

    x2 = min(x2, frame_w)
    y2 = min(y2, frame_h)
    src_x2 = src_x1 + (x2 - x1)
    src_y2 = src_y1 + (y2 - y1)

    if x2 <= x1 or y2 <= y1:
        return frame_bgr

    crop = overlay_bgra[src_y1:src_y2, src_x1:src_x2]
    alpha = crop[:, :, 3:4].astype(np.float32) / 255.0
    hat_bgr = crop[:, :, :3].astype(np.float32)
    roi = frame_bgr[y1:y2, x1:x2].astype(np.float32)
    frame_bgr[y1:y2, x1:x2] = (alpha * hat_bgr + (1.0 - alpha) * roi).astype(np.uint8)
    return frame_bgr


def draw_hats(frame: np.ndarray, registry, hat_images: dict, hat_assignment: dict,
              eliminated_ids: set = None, hat_identity_assignment: dict = None,
              auto_assign: bool = True) -> np.ndarray:
    """Draw one hat above each visible player using the player's tracked bounding box."""
    if not hat_images:
        return frame

    eliminated_ids = eliminated_ids or set()
    hat_identity_assignment = hat_identity_assignment if hat_identity_assignment is not None else {}
    hat_names = list(hat_images.keys())

    for tid, rec in registry.players.items():
        if rec.bbox is None or tid in eliminated_ids:
            continue

        if rec.name != "Unknown" and rec.name in hat_identity_assignment:
            hat_assignment[tid] = hat_identity_assignment[rec.name]

        if tid not in hat_assignment:
            if not auto_assign:
                continue
            hat_assignment[tid] = random.choice(hat_names)

        if rec.name != "Unknown" and rec.name not in hat_identity_assignment:
            hat_identity_assignment[rec.name] = hat_assignment[tid]

        hat_name = hat_assignment[tid]
        if hat_name not in hat_images:
            continue

        hat_img = hat_images[hat_name]
        x1, y1, x2, y2 = rec.bbox
        box_w = x2 - x1
        if box_w <= 0:
            continue

        hat_h, hat_w = hat_img.shape[:2]
        target_w = max(8, int(round(max(1, int(box_w * HAT_WIDTH_RATIO)) / 8.0) * 8))
        cache_key = (hat_name, target_w)
        resized_hat = _RESIZED_HAT_CACHE.get(cache_key)
        if resized_hat is None:
            scale = target_w / hat_w
            target_h = max(1, int(hat_h * scale))
            resized_hat = cv2.resize(hat_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
            _RESIZED_HAT_CACHE[cache_key] = resized_hat
        else:
            target_h = resized_hat.shape[0]

        center_x = (x1 + x2) // 2
        place_x = center_x - target_w // 2
        place_y = y1 - int(target_h * (1.0 - HAT_VERTICAL_OVERLAP_RATIO))
        frame = _overlay_rgba(frame, resized_hat, place_x, place_y)

    return frame


def cmd_hats_test(use_webcam: bool):
    """Run a standalone hat overlay test."""
    from .person_tracking import PersonTracker, PlayerRegistry, draw_registry

    if use_webcam:
        from .camera_utils import open_camera
        cap = open_camera()
        get_frame = lambda: cap.read()[1]
    else:
        from djitellopy import Tello
        tello = Tello()
        tello.connect()
        print(f"[INFO] Drone battery: {tello.get_battery()}%")
        tello.streamon()
        get_frame = lambda: cv2.cvtColor(tello.get_frame_read().frame, cv2.COLOR_RGB2BGR)

    tracker = PersonTracker()
    registry = PlayerRegistry()
    hat_images = load_hat_images()
    hat_assignment = {}

    if not hat_images:
        print("[ERROR] No hats were loaded. Add transparent PNG files to assets/hats/.")

    print("[INFO] Press 'q' to exit. Press 'r' to randomly reassign hats.")
    while True:
        frame = get_frame()
        if frame is None:
            continue

        track_ids, boxes = tracker.process(frame)
        now = registry.update(track_ids, boxes)
        frame = draw_registry(frame, registry, now)
        frame = draw_hats(frame, registry, hat_images, hat_assignment)

        cv2.imshow("Hat Overlay Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            hat_assignment.clear()

    cv2.destroyAllWindows()
    if use_webcam:
        cap.release()
    else:
        tello.streamoff()
