"""Motion baseline collection, labeling, evaluation, and live testing."""

import csv
import os
import time

import cv2
import numpy as np
import pandas as pd

from .config import (
    RAW_FRAMES_DIR,
    MOTION_LOG_CSV,
    CAPTURE_FPS,
    BLUR_KERNEL,
    MOTION_AREA_RATIO_THRESHOLD,
)
from .camera_utils import open_camera


def connect_tello():
    from djitellopy import Tello

    tello = Tello()
    tello.connect()
    print(f"[INFO] Drone battery: {tello.get_battery()}%")
    tello.streamon()
    return tello


def compute_motion_score(prev_gray, curr_gray):
    prev_blur = cv2.GaussianBlur(prev_gray, BLUR_KERNEL, 0)
    curr_blur = cv2.GaussianBlur(curr_gray, BLUR_KERNEL, 0)
    diff = cv2.absdiff(prev_blur, curr_blur)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    motion_ratio = np.count_nonzero(thresh) / thresh.size
    return motion_ratio, thresh


def cmd_collect(session_name: str, duration_sec: float, use_webcam: bool = False):
    session_dir = os.path.join(RAW_FRAMES_DIR, session_name)
    os.makedirs(session_dir, exist_ok=True)

    tello = None
    if use_webcam:
        print("[INFO] Using webcam mode for data collection.")
        cap = open_camera()
        get_frame = lambda: cap.read()[1]
    else:
        tello = connect_tello()
        get_frame = lambda: tello.get_frame_read().frame

    write_header = not os.path.exists(MOTION_LOG_CSV)
    csv_file = open(MOTION_LOG_CSV, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow(["session", "frame_idx", "timestamp", "filename", "motion_ratio", "label"])

    prev_gray = None
    frame_interval = 1.0 / CAPTURE_FPS
    start_time = time.time()
    frame_idx = 0

    print(
        f"[INFO] Starting session '{session_name}' for {duration_sec} seconds "
        f"at {CAPTURE_FPS} frame(s)/second."
    )
    print("[INFO] Press 'q' in the preview window to stop early.")

    try:
        while time.time() - start_time < duration_sec:
            loop_start = time.time()
            frame = get_frame()
            if frame is None:
                continue

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if not use_webcam else frame
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            motion_ratio = 0.0
            if prev_gray is not None:
                motion_ratio, diff_vis = compute_motion_score(prev_gray, gray)
                cv2.imshow("motion diff (debug)", diff_vis)

            filename = f"frame_{frame_idx:05d}.jpg"
            filepath = os.path.join(session_dir, filename)
            cv2.imwrite(filepath, frame_bgr)

            writer.writerow([session_name, frame_idx, time.time(), filename, f"{motion_ratio:.5f}", ""])
            csv_file.flush()

            cv2.putText(
                frame_bgr,
                f"motion={motion_ratio:.3f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            cv2.imshow("Tello Data Collection", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            prev_gray = gray
            frame_idx += 1
            elapsed = time.time() - loop_start
            time.sleep(max(0, frame_interval - elapsed))
    finally:
        csv_file.close()
        if tello:
            tello.streamoff()
        else:
            cap.release()
        cv2.destroyAllWindows()
        print(f"[INFO] Saved {frame_idx} frame(s) to {session_dir}")
        print(f"[INFO] Motion log updated: {MOTION_LOG_CSV}")


def cmd_label(session_name: str, label: str):
    df = pd.read_csv(MOTION_LOG_CSV)
    mask = df["session"] == session_name
    if mask.sum() == 0:
        print(f"[WARN] No rows found for session '{session_name}'.")
        return
    df.loc[mask, "label"] = label
    df.to_csv(MOTION_LOG_CSV, index=False)
    print(f"[INFO] Labeled {mask.sum()} frame(s) from '{session_name}' as '{label}'.")


def motion_predict(motion_ratio: float, threshold: float = MOTION_AREA_RATIO_THRESHOLD) -> str:
    return "moving" if motion_ratio >= threshold else "still"


def cmd_evaluate(threshold: float = MOTION_AREA_RATIO_THRESHOLD):
    df = pd.read_csv(MOTION_LOG_CSV)
    df = df.dropna(subset=["label"])
    df = df[df["label"].isin(["still", "moving"])]

    if len(df) == 0:
        print("[WARN] No labeled data found. Run: main.py label ...")
        return

    df["pred"] = df["motion_ratio"].apply(lambda r: motion_predict(r, threshold))
    tp = ((df["label"] == "moving") & (df["pred"] == "moving")).sum()
    tn = ((df["label"] == "still") & (df["pred"] == "still")).sum()
    fp = ((df["label"] == "still") & (df["pred"] == "moving")).sum()
    fn = ((df["label"] == "moving") & (df["pred"] == "still")).sum()

    total = len(df)
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print("===== Baseline Motion Detector Evaluation =====")
    print(f"threshold (motion_ratio) = {threshold}")
    print(f"evaluated frames         = {total}")
    print(f"Accuracy  = {accuracy:.3f}")
    print(f"Precision = {precision:.3f}")
    print(f"Recall    = {recall:.3f}")
    print(f"F1-score  = {f1:.3f}")
    print(f"Confusion: TP={tp} TN={tn} FP={fp} FN={fn}")
    print("===============================================")


def cmd_sweep():
    df = pd.read_csv(MOTION_LOG_CSV).dropna(subset=["label"])
    df = df[df["label"].isin(["still", "moving"])]
    if len(df) == 0:
        print("[WARN] No labeled data found.")
        return

    best = (0, -1)
    for threshold in np.arange(0.001, 0.05, 0.001):
        pred = df["motion_ratio"].apply(lambda r: motion_predict(r, threshold))
        tp = ((df["label"] == "moving") & (pred == "moving")).sum()
        fp = ((df["label"] == "still") & (pred == "moving")).sum()
        fn = ((df["label"] == "moving") & (pred == "still")).sum()
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        if f1 > best[1]:
            best = (threshold, f1)

    print(f"[INFO] Best approximate threshold = {best[0]:.4f} (F1={best[1]:.3f})")
    print("Update MOTION_AREA_RATIO_THRESHOLD in config.py with this value if desired.")


def cmd_motion_live(use_webcam: bool = True, threshold: float = MOTION_AREA_RATIO_THRESHOLD):
    if use_webcam:
        cap = open_camera()
        get_frame = lambda: cap.read()[1]
    else:
        tello = connect_tello()
        get_frame = lambda: cv2.cvtColor(tello.get_frame_read().frame, cv2.COLOR_RGB2BGR)

    prev_gray = None
    print("[INFO] Press 'q' to exit.")
    while True:
        frame = get_frame()
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray_blur)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            ratio = np.count_nonzero(thresh) / thresh.size
            label = motion_predict(ratio, threshold)
            color = (0, 0, 255) if label == "moving" else (0, 200, 0)
            cv2.putText(frame, f"{label} ({ratio:.3f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Baseline Motion Detector", frame)
        prev_gray = gray_blur
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    if use_webcam:
        cap.release()
    else:
        tello.streamoff()
