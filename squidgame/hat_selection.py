"""Pre-game visual interface where every confirmed player chooses one hat."""

import math
import os
import time

import cv2
import numpy as np

from .config import FACE_ID_INTERVAL_FRAMES
from .hat_overlay import draw_hats
from .person_tracking import draw_registry

WINDOW_NAME = "Squid Game Drone"
_THUMB_CACHE = {}


def _alpha_blit(dst_bgr: np.ndarray, src_bgra: np.ndarray, x: int, y: int) -> None:
    sh, sw = src_bgra.shape[:2]
    dh, dw = dst_bgr.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(dw, x + sw), min(dh, y + sh)
    if x2 <= x1 or y2 <= y1:
        return
    sx1, sy1 = x1 - x, y1 - y
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
    crop = src_bgra[sy1:sy2, sx1:sx2]
    alpha = crop[:, :, 3:4].astype(np.float32) / 255.0
    roi = dst_bgr[y1:y2, x1:x2].astype(np.float32)
    dst_bgr[y1:y2, x1:x2] = (alpha * crop[:, :, :3] + (1.0 - alpha) * roi).astype(np.uint8)


def _fit_hat(hat_bgra: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = hat_bgra.shape[:2]
    if h <= 0 or w <= 0:
        return hat_bgra
    scale = max(0.01, min(max_w / w, max_h / h))
    return cv2.resize(
        hat_bgra,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _pretty_hat_name(filename: str) -> str:
    return os.path.splitext(filename)[0].replace("_", " ").title()


def _resolve_active_player(registry, player_info: dict):
    tid = player_info["track_id"]
    rec = registry.players.get(tid)
    if rec is not None:
        if rec.name != "Unknown":
            player_info["name"] = rec.name
        return rec

    name = player_info.get("name", "Unknown")
    if name != "Unknown":
        for current_tid, current_rec in registry.players.items():
            if current_rec.name == name:
                player_info["track_id"] = current_tid
                return current_rec
    return None


def _build_selection_canvas(frame, registry, now, hat_images, hat_assignment,
                            hat_by_name, active_rec, player_number, total_players):
    preview = draw_registry(frame.copy(), registry, now)
    preview = draw_hats(
        preview,
        registry,
        hat_images,
        hat_assignment,
        hat_identity_assignment=hat_by_name,
        auto_assign=False,
    )

    if active_rec is not None and active_rec.bbox is not None:
        x1, y1, x2, y2 = active_rec.bbox
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 215, 255), 4)

    fh, fw = preview.shape[:2]
    hat_names = list(hat_images.keys())
    cols = min(4, max(1, len(hat_names)))
    rows = max(1, math.ceil(len(hat_names) / cols))
    panel_h = 92 + rows * 150 + 28
    canvas = np.full((fh + panel_h, fw, 3), 24, dtype=np.uint8)
    canvas[:fh] = preview

    panel_y = fh
    cv2.rectangle(canvas, (0, panel_y), (fw, fh + panel_h), (24, 24, 24), -1)
    cv2.line(canvas, (0, panel_y), (fw, panel_y), (80, 80, 80), 2)

    if active_rec is not None:
        label = active_rec.name if active_rec.name != "Unknown" else f"ID {active_rec.track_id}"
        title = f"PLAYER {player_number}/{total_players}: {label} - CHOOSE YOUR HAT"
    else:
        title = f"PLAYER {player_number}/{total_players} - RETURN TO CAMERA"

    cv2.putText(canvas, title, (22, panel_y + 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Click a hat or press its number. Q = cancel", (22, panel_y + 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (185, 185, 185), 1, cv2.LINE_AA)

    gap = 14
    outer_margin = 18
    card_w = max(120, (fw - outer_margin * 2 - gap * (cols - 1)) // cols)
    card_h = 134
    rects = []

    for idx, fname in enumerate(hat_names):
        row, col = divmod(idx, cols)
        x = outer_margin + col * (card_w + gap)
        y = panel_y + 82 + row * 150
        x2, y2 = min(fw - outer_margin, x + card_w), y + card_h

        cv2.rectangle(canvas, (x, y), (x2, y2), (48, 48, 48), -1)
        cv2.rectangle(canvas, (x, y), (x2, y2), (105, 105, 105), 2)

        thumb_key = (fname, max(50, x2 - x - 28), 78)
        thumb = _THUMB_CACHE.get(thumb_key)
        if thumb is None:
            thumb = _fit_hat(hat_images[fname], thumb_key[1], thumb_key[2])
            _THUMB_CACHE[thumb_key] = thumb
        th, tw = thumb.shape[:2]
        _alpha_blit(canvas, thumb, x + (x2 - x - tw) // 2, y + 10)

        text = f"{idx + 1}. {_pretty_hat_name(fname)}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
        tx = x + max(8, (x2 - x - text_size[0]) // 2)
        cv2.putText(canvas, text, (tx, y2 - 13), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (240, 240, 240), 1, cv2.LINE_AA)
        rects.append((x, y, x2, y2, fname))

    return canvas, rects


def run_hat_selection_phase(drone, tracker, registry, hat_images,
                            face_identifier=None, hat_assignment=None,
                            hat_by_name=None) -> bool:
    """Let each confirmed player select exactly one hat before the round starts."""
    if not hat_images:
        print("[HATS] No hats are available. Continuing without hat selection.")
        return True

    hat_assignment = hat_assignment if hat_assignment is not None else {}
    hat_by_name = hat_by_name if hat_by_name is not None else {}

    now = time.time()
    confirmed = registry.confirmed_players(now)
    if not confirmed:
        print("[HATS] No confirmed players are available for hat selection.")
        return True

    confirmed.sort(key=lambda r: ((r.bbox[0] + r.bbox[2]) / 2) if r.bbox else r.track_id)
    players = [{"track_id": r.track_id, "name": r.name} for r in confirmed]
    hat_names = list(hat_images.keys())

    print(f"[HATS] Starting hat selection for {len(players)} player(s).")
    print("[HATS] Choose with the mouse or number keys. The choice stays locked for the session.")

    click_state = {"point": None}

    def _mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_state["point"] = (x, y)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except cv2.error:
        pass
    cv2.setMouseCallback(WINDOW_NAME, _mouse)

    player_idx = 0
    frame_count = 0

    while player_idx < len(players):
        frame = drone.get_frame()
        if frame is None:
            continue
        frame_count += 1

        track_ids, boxes = tracker.process(frame)
        now = registry.update(track_ids, boxes)

        if face_identifier is not None and frame_count % FACE_ID_INTERVAL_FRAMES == 0:
            box_map = dict(zip(track_ids, boxes))
            names = face_identifier.identify_players(frame, box_map)
            for tid, name in names.items():
                if tid in registry.players:
                    registry.players[tid].name = name

        active_info = players[player_idx]
        active_rec = _resolve_active_player(registry, active_info)

        canvas, rects = _build_selection_canvas(
            frame, registry, now, hat_images, hat_assignment, hat_by_name,
            active_rec, player_idx + 1, len(players),
        )
        cv2.imshow(WINDOW_NAME, canvas)

        selected = None
        point = click_state.pop("point", None)
        click_state["point"] = None
        if point is not None and active_rec is not None:
            px, py = point
            for x1, y1, x2, y2, fname in rects:
                if x1 <= px <= x2 and y1 <= py <= y2:
                    selected = fname
                    break

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[HATS] Hat selection cancelled.")
            cv2.destroyWindow(WINDOW_NAME)
            return False
        if active_rec is not None and ord("1") <= key <= ord("9"):
            choice_idx = key - ord("1")
            if choice_idx < len(hat_names):
                selected = hat_names[choice_idx]

        if selected is not None and active_rec is not None:
            tid = active_rec.track_id
            hat_assignment[tid] = selected
            if active_rec.name != "Unknown":
                hat_by_name[active_rec.name] = selected
                label = active_rec.name
            else:
                label = f"ID {tid}"
            print(f"[HATS] {label} selected {_pretty_hat_name(selected)}")
            player_idx += 1

    print("[HATS] All hats are locked. Press ENTER or SPACE to start, or Q to cancel.")
    while True:
        frame = drone.get_frame()
        if frame is None:
            continue
        track_ids, boxes = tracker.process(frame)
        now = registry.update(track_ids, boxes)

        if face_identifier is not None:
            frame_count += 1
            if frame_count % FACE_ID_INTERVAL_FRAMES == 0:
                box_map = dict(zip(track_ids, boxes))
                names = face_identifier.identify_players(frame, box_map)
                for tid, name in names.items():
                    if tid in registry.players:
                        registry.players[tid].name = name

        preview = draw_registry(frame.copy(), registry, now)
        preview = draw_hats(
            preview,
            registry,
            hat_images,
            hat_assignment,
            hat_identity_assignment=hat_by_name,
            auto_assign=False,
        )
        cv2.rectangle(preview, (0, 0), (preview.shape[1], 52), (25, 25, 25), -1)
        cv2.putText(preview, "HATS LOCKED - ENTER / SPACE TO START", (18, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.destroyWindow(WINDOW_NAME)
            return False
        if key in (13, 10, ord(" ")):
            break

    cv2.destroyWindow(WINDOW_NAME)
    return True
