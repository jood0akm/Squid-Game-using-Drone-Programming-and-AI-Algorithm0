"""Face recognition, cap-color fallback, and player registration commands."""

import os

import cv2
import numpy as np

from .config import (
    CAP_COLOR_HSV_RANGES,
    CAP_COLOR_FILE,
    CAP_COLOR_MIN_RATIO,
    HEAD_REGION_RATIO,
    FACE_MODEL_NAME,
    FACE_CTX_ID,
    FACE_DET_SIZE,
    FACE_MATCH_THRESHOLD,
    PLAYERS_FILE,
    REGISTER_TARGET_COUNT,
)
from .camera_utils import open_camera
from .storage import load_players, save_players, load_cap_colors, save_cap_colors


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-6)
    b_norm = b / (np.linalg.norm(b) + 1e-6)
    return float(np.dot(a_norm, b_norm))


def detect_cap_color(frame_bgr, bbox: tuple):
    """Return the dominant configured cap color in the upper head region."""
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    if h <= 0 or x2 <= x1:
        return None

    head_y1 = max(0, y1 - int(h * 0.15))
    head_y2 = y1 + int(h * HEAD_REGION_RATIO)
    region = frame_bgr[head_y1:head_y2, x1:x2]
    if region.size == 0:
        return None

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    best_color, best_ratio = None, 0.0
    for color, ranges in CAP_COLOR_HSV_RANGES.items():
        mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
            mask_total = cv2.bitwise_or(mask_total, mask)
        ratio = np.count_nonzero(mask_total) / mask_total.size
        if ratio > best_ratio:
            best_ratio, best_color = ratio, color

    return best_color if best_ratio >= CAP_COLOR_MIN_RATIO else None


class FaceIdentifier:
    """Match detected faces to registered embeddings from players.json."""

    def __init__(self):
        from insightface.app import FaceAnalysis

        print("[INFO] Loading face recognition model...")
        self.app = FaceAnalysis(name=FACE_MODEL_NAME)
        self.app.prepare(ctx_id=FACE_CTX_ID, det_size=FACE_DET_SIZE)
        print(f"[DEBUG] Reading faces from: {os.path.abspath(PLAYERS_FILE)}")
        self.known = self._load_known_faces()
        if self.known:
            print(f"[INFO] Loaded {len(self.known)} registered face(s): {', '.join(n for n, _ in self.known)}")

    @staticmethod
    def _load_known_faces():
        players = load_players()
        return [(name, np.array(emb, dtype=np.float32)) for name, emb in players.items()]

    def match_name(self, embedding: np.ndarray) -> str:
        if not self.known:
            return "Unknown"
        best_name, best_score = "Unknown", -1.0
        for name, known_emb in self.known:
            score = cosine_similarity(embedding, known_emb)
            if score > best_score:
                best_score, best_name = score, name
        return best_name if best_score >= FACE_MATCH_THRESHOLD else "Unknown"

    def identify_players(self, frame_bgr, player_boxes: dict) -> dict:
        """Map each track ID to a registered name when a matching face is found."""
        faces = self.app.get(frame_bgr)
        result = {}
        for tid, box in player_boxes.items():
            x1, y1, x2, y2 = box
            name = "Unknown"
            for face in faces:
                fb = face.bbox.astype(int)
                fcx, fcy = (fb[0] + fb[2]) // 2, (fb[1] + fb[3]) // 2
                if x1 <= fcx <= x2 and y1 <= fcy <= y2:
                    name = self.match_name(face.embedding)
                    break
            result[tid] = name
        return result


def cmd_register(name: str = None, use_webcam: bool = True, cap_color: str = None):
    """Register a player using up to REGISTER_TARGET_COUNT face captures."""
    if not name:
        name = input("Enter player name: ").strip()

    if not name:
        print("[WARN] Player name is required.")
        return

    from insightface.app import FaceAnalysis

    if cap_color and cap_color not in CAP_COLOR_HSV_RANGES:
        print(
            f"[WARN] Unsupported cap color '{cap_color}'. Available colors: "
            f"{', '.join(CAP_COLOR_HSV_RANGES.keys())}"
        )
        return

    print("[INFO] Loading the face model. The first run may take longer...")
    app = FaceAnalysis(name=FACE_MODEL_NAME)
    app.prepare(ctx_id=FACE_CTX_ID, det_size=FACE_DET_SIZE)

    if use_webcam:
        cap = open_camera()
    else:
        print("[ERROR] Registration currently supports webcam mode only. Add --webcam.")
        return

    if not cap.isOpened():
        print("[ERROR] Could not open the webcam. Close other apps that may be using it.")
        return

    players = load_players()
    embeddings = []
    print(f"[INFO] {name}, face the camera. Press SPACE to capture each photo, or Q to exit.")

    while len(embeddings) < REGISTER_TARGET_COUNT:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        faces = app.get(frame)
        display = frame.copy()

        if faces:
            box = faces[0].bbox.astype(int)
            cv2.rectangle(display, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        cv2.putText(
            display,
            f"Captured: {len(embeddings)}/{REGISTER_TARGET_COUNT}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Face Registration", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and faces:
            embeddings.append(faces[0].embedding)
            avg_so_far = np.mean(embeddings, axis=0).astype(np.float32)
            players[name] = avg_so_far.tolist()
            save_players(players)
            print(
                f"Captured and saved {len(embeddings)}/{REGISTER_TARGET_COUNT} "
                f"to {os.path.basename(PLAYERS_FILE)}"
            )
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not embeddings:
        print("[WARN] No face images were captured. Try again.")
        return

    print(f"[INFO] Saved '{name}' successfully with {len(embeddings)} capture(s) in {PLAYERS_FILE}.")

    if cap_color:
        cap_colors = load_cap_colors()
        cap_colors[name] = cap_color
        save_cap_colors(cap_colors)
        print(f"[INFO] Linked '{name}' to cap color '{cap_color}' in {CAP_COLOR_FILE}.")


def cmd_list_players():
    """Print all registered players and optional cap colors."""
    print(f"[DEBUG] Reading from: {os.path.abspath(PLAYERS_FILE)}")
    players = load_players()
    cap_colors = load_cap_colors()
    if not players:
        print("[INFO] No registered players yet. Use: register --name <name> --webcam")
        return
    print(f"[INFO] Registered players ({len(players)}):")
    for name in sorted(players.keys()):
        color = cap_colors.get(name)
        suffix = f" (cap: {color})" if color else ""
        print(f"  - {name}{suffix}")
