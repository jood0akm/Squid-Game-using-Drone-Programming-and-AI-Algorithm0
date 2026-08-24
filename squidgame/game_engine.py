"""English documentation."""

import os
import random
import time
import threading
from enum import Enum

import cv2
import numpy as np

from .config import (
    ROTATE_DEGREES,
    CONFIRM_SECONDS,
    FACE_ID_INTERVAL_FRAMES,
    REQUIRED_CONFIRMED_PLAYERS,
    GREEN_LIGHT_MIN_SEC,
    GREEN_LIGHT_MAX_SEC,
    RED_LIGHT_MIN_SEC,
    RED_LIGHT_MAX_SEC,
    RED_LIGHT_GRACE_PERIOD_SEC,
    PROXIMITY_WIN_RATIO,
    PROXIMITY_WIN_GRACE_SEC,
    VOICE_LANG_GREEN,
    VOICE_LANG_RED,
    SACK_RACE_TIME_MULTIPLIER,
    FINISH_LINE_Y_RATIO,
    EDGE_WARNING_MARGIN_RATIO,
    EDGE_WARNING_COOLDOWN_SEC,
    EVIDENCE_DIR,
    QUICK_TURN_ENABLED,
    QUICK_TURN_YAW_SPEED,
    QUICK_TURN_BURST_SEC,
    HAS_TELLO_TALENT,
    MOTION_CONFIRM_FRAMES,
    HAT_OVERLAY_ENABLED,
    RED_EYE_FULLSCREEN_SEC,
    MOTION_SAMPLE_INTERVAL_SEC,
    HOVER_STABILIZER_ENABLED,
    HOVER_STABILIZER_HZ,
    HOVER_SETTLE_SEC,
    HOVER_SPEED_DEADBAND_CMS,
    HOVER_SPEED_GAIN,
    HOVER_SPEED_MAX_CORRECTION,
    VIDEO_FREEZE_CHECK_ENABLED,
    VIDEO_FREEZE_DIFF_THRESHOLD,
    VIDEO_FREEZE_SECONDS,
    VIDEO_FREEZE_SAMPLE_INTERVAL_SEC,
    TARGET_HOVER_HEIGHT_ENABLED,
    TARGET_HOVER_HEIGHT_CM,
    MOVE_COMMAND_MIN_CM,
    FINE_TUNE_HEIGHT_ENABLED,
    FINE_TUNE_TOLERANCE_CM,
    FINE_TUNE_MAX_ITERATIONS,
    FINE_TUNE_PULSE_SEC,
    FINE_TUNE_SPEED,
    FINE_TUNE_SETTLE_SEC,
    FINE_TUNE_HARD_CEILING_CM,
)
from .storage import load_cap_colors
from .camera_utils import open_camera
from .voice import VoiceAnnouncer
from .leaderboard import update_leaderboard
from .face_id import FaceIdentifier, detect_cap_color
from .safety import SafetyMonitor, SAFETY_MIN_BATTERY_TAKEOFF
from .hat_overlay import load_hat_images, draw_hats
from .hat_selection import run_hat_selection_phase
from .traffic_light import TrafficLightController
from .person_tracking import (
    PersonTracker,
    PlayerRegistry,
    PerPlayerMotionChecker,
    draw_registry,
)
from .dashboard import (
    draw_battery,
    draw_player_list,
    draw_pixel_logo,
    draw_fullscreen_eyes,
    draw_results_dashboard,
    build_live_entries,
    play_beep,
    COLOR_ALIVE,
    COLOR_OUT,
    COLOR_NEW,
)
from .config import RESULTS_DISPLAY_SEC, GAME_OVER_POLL_MS, AUTO_NEXT_ROUND, NEXT_ROUND_DELAY_SEC


class LightState(Enum):
    LOBBY = "lobby"
    GREEN = "green"
    RED = "red"


class HoverStabilizer:
    """Background hover-assist loop, run while the drone is airborne.

    Reads Tello telemetry and sends small corrective rc_control nudges for
    HORIZONTAL drift only:
      - vgx/vgy (get_speed_x/get_speed_y): fused IMU + downward optical-flow
        velocity — used to detect and gently counter horizontal drift

    Height (barometer) and pitch/roll (IMU) are read for logging only and are
    NEVER used to command vertical movement here.

    ⚠️ SAFETY NOTE: an earlier version of this class also auto-corrected
    altitude (up/down) using the barometer reading. During real testing this
    caused the drone to climb into the ceiling — the Tello has NO upward
    obstacle sensor, so nothing stops a software altitude-hold loop from
    flying it straight into whatever is above it if the reading or timing is
    even slightly off. That vertical-correction code has been removed
    entirely, not just disabled, so it cannot accidentally come back. Do not
    re-add automatic vertical rc_control here without a hard ceiling-distance
    safety check (e.g. a downward/upward range sensor) and in-person testing
    with a spotter ready to catch the drone.

    This class is disabled by default (HOVER_STABILIZER_ENABLED = False in
    config.py). The Tello already self-stabilizes attitude and altitude
    internally; only enable this for extra horizontal drift correction, and
    test it first in an open space with a high ceiling.
    """

    def __init__(self, tello, turn_done_event: threading.Event, turn_lock: threading.Lock):
        self._tello = tello
        self._turn_done = turn_done_event
        self._turn_lock = turn_lock
        self._running = False
        self._thread = None
        self._start_time = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="hover-stabilizer")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _loop(self):
        period = 1.0 / max(0.5, HOVER_STABILIZER_HZ)
        while self._running:
            time.sleep(period)
            if not self._running:
                break

            # Never fight an in-progress 180-degree rotation command.
            if not self._turn_done.is_set():
                continue

            if time.time() - self._start_time < HOVER_SETTLE_SEC:
                continue

            try:
                # Height/pitch/roll are logged only — never used for correction here.
                _height = self._tello.get_height()   # barometer (diagnostics only)
                _pitch = self._tello.get_pitch()      # IMU (diagnostics only)
                _roll = self._tello.get_roll()        # IMU (diagnostics only)
                vgx = self._tello.get_speed_x()       # fused IMU + optical-flow ("third" sensor)
                vgy = self._tello.get_speed_y()
            except Exception:
                continue

            # Skip a correction cycle if a rotation started while we were reading telemetry.
            if not self._turn_done.is_set():
                continue

            lr, fb = 0, 0

            if abs(vgx) > HOVER_SPEED_DEADBAND_CMS:
                fb = int(np.clip(-vgx * HOVER_SPEED_GAIN,
                                  -HOVER_SPEED_MAX_CORRECTION, HOVER_SPEED_MAX_CORRECTION))
            if abs(vgy) > HOVER_SPEED_DEADBAND_CMS:
                lr = int(np.clip(-vgy * HOVER_SPEED_GAIN,
                                  -HOVER_SPEED_MAX_CORRECTION, HOVER_SPEED_MAX_CORRECTION))

            if lr or fb:
                try:
                    self._tello.send_rc_control(lr, fb, 0, 0)  # up/down always 0 — see class docstring
                except Exception:
                    pass


class DroneController:
    """Controls Tello/webcam capture and keeps flight commands off the video loop."""

    def __init__(self, use_webcam: bool, allow_flight: bool):
        self.use_webcam = use_webcam
        self.allow_flight = allow_flight
        self.tello = None
        self.cap = None
        self.frame_read = None
        self.airborne = False

        # The drone must be placed facing the players before starting.
        self.facing_players = True
        self._turn_thread = None
        self._turn_done = threading.Event()
        self._turn_done.set()
        self._turn_error = None
        self._turn_target_players = None
        self._turn_lock = threading.Lock()
        self._stabilizer = None

        # Tello safety feature: the aircraft auto-lands if it receives no SDK
        # command input for ~15 seconds. Keep a zero-velocity RC heartbeat
        # running for the whole airborne session (lobby, game, results, restart).
        # The heartbeat pauses while an exact 180-degree SDK rotation is active
        # so it cannot interfere with that discrete command.
        self._hover_keepalive_stop = threading.Event()
        self._hover_keepalive_thread = None
        self._hover_keepalive_interval_sec = 3.0

        # Video-freeze detection state (drone mode only).
        self._freeze_last_gray = None
        self._freeze_last_change_time = time.time()
        self._freeze_last_sample_at = 0.0

        if use_webcam:
            self.cap = open_camera()
            if not self.cap.isOpened():
                raise RuntimeError("Could not open the webcam. Close other apps that may be using it.")
            return

        from djitellopy import Tello

        self.tello = Tello()
        print("[DRONE] Connecting to Tello...")
        self.tello.connect()
        print(f"[DRONE] Battery: {self.tello.get_battery()}%")

        try:
            self.tello.streamoff()
        except Exception:
            pass

        # Keep 720p/30 FPS quality, but cap bitrate to reduce Wi-Fi buffering/latency.
        try:
            self.tello.set_video_resolution(Tello.RESOLUTION_720P)
            self.tello.set_video_fps(Tello.FPS_30)
            self.tello.set_video_bitrate(Tello.BITRATE_3MBPS)
            print("[VIDEO] Low-latency mode: 720p / 30 FPS / 3 Mbps")
        except Exception as exc:
            print(f"[VIDEO] Could not apply video settings: {exc}")

        self.tello.streamon()

        # Give Tello a moment to begin sending H.264 packets before PyAV opens
        # UDP 11111. On some Windows/Wi-Fi setups opening immediately after
        # streamon intermittently raises "Failed to grab video frames".
        time.sleep(1.0)
        last_stream_error = None
        for attempt in range(1, 4):
            try:
                # with_queue=False means we always read the newest decoded frame,
                # never an old queued frame.
                self.frame_read = self.tello.get_frame_read(with_queue=False)
                time.sleep(0.8)
                print(f"[VIDEO] Stream reader ready (attempt {attempt}/3).")
                break
            except Exception as exc:
                last_stream_error = exc
                self.frame_read = None
                print(f"[VIDEO] Stream open attempt {attempt}/3 failed: {exc}")
                if attempt < 3:
                    try:
                        self.tello.streamoff()
                    except Exception:
                        pass
                    time.sleep(0.5)
                    self.tello.streamon()
                    time.sleep(1.2)
        else:
            raise RuntimeError(
                "Could not open Tello video stream after 3 attempts. "
                "Close any other program using UDP port 11111, reconnect to the Tello Wi-Fi, "
                f"then try again. Last error: {last_stream_error}"
            )

    def _start_hover_keepalive(self):
        """Prevent Tello's 15-second no-command auto-land while hovering.

        Uses the documented zero-velocity RC command instead of relying only
        on the optional 'keepalive' command. This thread stays active from
        takeoff until Q/landing, including lobby and results screens.
        """
        if self.use_webcam or not self.allow_flight or self.tello is None:
            return
        if self._hover_keepalive_thread is not None and self._hover_keepalive_thread.is_alive():
            return

        self._hover_keepalive_stop.clear()

        def _loop():
            print(f"[DRONE] Hover keepalive started (RC zero every {self._hover_keepalive_interval_sec:.0f}s).")
            while not self._hover_keepalive_stop.is_set():
                # Do not overlap an exact cw/ccw command with an RC command.
                if self.airborne and self._turn_done.is_set():
                    try:
                        self.tello.send_rc_control(0, 0, 0, 0)
                    except Exception as exc:
                        print(f"[DRONE] Hover keepalive warning: {exc}")

                # Event.wait lets landing stop the thread immediately rather
                # than waiting for time.sleep() to finish.
                if self._hover_keepalive_stop.wait(self._hover_keepalive_interval_sec):
                    break

            print("[DRONE] Hover keepalive stopped.")

        self._hover_keepalive_thread = threading.Thread(
            target=_loop, daemon=True, name="tello-hover-keepalive"
        )
        self._hover_keepalive_thread.start()

    def _stop_hover_keepalive(self):
        self._hover_keepalive_stop.set()
        thread = self._hover_keepalive_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._hover_keepalive_thread = None

    def get_frame(self):
        if self.use_webcam:
            ok, frame = self.cap.read()
            return frame if ok else None

        if self.frame_read is None:
            return None

        frame = self.frame_read.frame
        if frame is None:
            return None

        # DJITelloPy/PyAV provides RGB; OpenCV drawing expects BGR.
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self._check_video_freeze(bgr)
        return bgr

    def _check_video_freeze(self, frame_bgr):
        """Detect a Wi-Fi video stall (the same frame repeating) and recover
        automatically. Without this, the game silently "sees" a still image —
        no motion is ever detected and no proximity win ever triggers, which
        looks like the game simply isn't working, even though everything else
        is running fine."""
        if not VIDEO_FREEZE_CHECK_ENABLED:
            return

        now = time.time()
        if now - self._freeze_last_sample_at < VIDEO_FREEZE_SAMPLE_INTERVAL_SEC:
            return
        self._freeze_last_sample_at = now

        small = cv2.resize(frame_bgr, (80, 45), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if self._freeze_last_gray is not None:
            diff = float(cv2.absdiff(gray, self._freeze_last_gray).mean())
            if diff > VIDEO_FREEZE_DIFF_THRESHOLD:
                self._freeze_last_change_time = now
            elif now - self._freeze_last_change_time > VIDEO_FREEZE_SECONDS:
                print("[VIDEO] Feed looks frozen (Wi-Fi stall). Restarting the video stream...")
                self._recover_video_stream()
                self._freeze_last_change_time = now  # avoid an immediate retrigger

        self._freeze_last_gray = gray

    def _recover_video_stream(self):
        try:
            self.tello.streamoff()
            time.sleep(0.3)
            self.tello.streamon()
            self.frame_read = self.tello.get_frame_read(with_queue=False)
            time.sleep(0.5)
            print("[VIDEO] Stream restarted.")
        except Exception as exc:
            print(f"[VIDEO] Stream restart failed: {exc}")

    def get_battery(self):
        if self.use_webcam or self.tello is None:
            return None
        try:
            return self.tello.get_battery()
        except Exception:
            return None

    def takeoff(self) -> bool:
        """Take off once. Repeated calls while airborne do nothing."""
        if self.airborne:
            return True

        if self.use_webcam or not self.allow_flight:
            print("[SIM] takeoff (no real flight)")
            self.airborne = True
            return True

        try:
            battery = self.tello.get_battery()
            if battery < SAFETY_MIN_BATTERY_TAKEOFF:
                print(
                    f"[SAFETY] Battery {battery}% is below safe limit "
                    f"({SAFETY_MIN_BATTERY_TAKEOFF}%)."
                )
                return False

            print("[DRONE] Taking off before player recognition...")
            self.tello.takeoff()
            self.airborne = True
            print("[DRONE] Airborne. Player recognition will run while hovering.")

            self._climb_to_target_height()

            # From this point until Q/landing, keep sending a harmless zero-RC
            # command so the Tello never reaches its 15-second no-command
            # auto-land timeout while we are recognizing players or showing results.
            self._start_hover_keepalive()

            # Open the screen the instant the target height is reached — no
            # extra steps in between.
            _set_fullscreen()

            if HOVER_STABILIZER_ENABLED:
                self._stabilizer = HoverStabilizer(self.tello, self._turn_done, self._turn_lock)
                self._stabilizer.start()
                print("[STABILIZER] Hover-assist loop started (barometer + IMU + optical-flow speed).")

            return True
        except Exception as exc:
            print(f"[DRONE] Takeoff failed: {exc}")
            self.airborne = False
            return False

    def _climb_to_target_height(self):
        """Climb immediately after takeoff to TARGET_HOVER_HEIGHT_CM using a
        single discrete move_up() SDK command (not a continuous correction
        loop), fired right away so the ascent reads as one continuous motion
        instead of a stop-then-climb. After this one command, the Tello's own
        internal barometer/IMU/optical-flow hold keeps it there, exactly like
        it does at its default takeoff height. A short, bounded fine-tuning
        pass then closes the remaining gap below the SDK's 20cm move floor."""
        if not TARGET_HOVER_HEIGHT_ENABLED:
            return
        try:
            current_height = self.tello.get_height()
            delta = TARGET_HOVER_HEIGHT_CM - current_height

            if delta >= MOVE_COMMAND_MIN_CM:
                delta = min(delta, 500)  # SDK move command upper bound
                print(f"[DRONE] Climbing {delta} cm to reach target height (~{TARGET_HOVER_HEIGHT_CM} cm)...")
                self.tello.move_up(delta)
            else:
                print(f"[DRONE] Already near target height ({current_height} cm). Fine-tuning only.")

            self._fine_tune_height()
        except Exception as exc:
            print(f"[DRONE] Could not climb to target height: {exc}")

    def _fine_tune_height(self):
        """Bounded, defensive fine-tuning pass to close the gap the discrete
        move_up() command can't reach (it has a hard 20cm minimum). This is
        deliberately NOT a continuous correction loop:
          - each correction is a short pulse (FINE_TUNE_PULSE_SEC), followed
            by an explicit full stop and a real height re-measurement before
            deciding whether to continue
          - a hard ceiling (FINE_TUNE_HARD_CEILING_CM) can never be crossed
            by an upward pulse, independent of anything else
          - at most FINE_TUNE_MAX_ITERATIONS pulses total — it always gives
            up and stops after that, even if not perfectly on target
        """
        if not FINE_TUNE_HEIGHT_ENABLED:
            return

        for i in range(FINE_TUNE_MAX_ITERATIONS):
            try:
                current = self.tello.get_height()
            except Exception:
                return

            error = TARGET_HOVER_HEIGHT_CM - current

            if abs(error) <= FINE_TUNE_TOLERANCE_CM:
                print(f"[DRONE] Fine-tune complete: {current} cm (target {TARGET_HOVER_HEIGHT_CM} cm).")
                return

            going_up = error > 0

            # Hard safety ceiling: refuse any upward pulse at/above the cap, full stop.
            if going_up and current >= FINE_TUNE_HARD_CEILING_CM:
                print(f"[SAFETY] Fine-tune stopped — at hard height ceiling ({current} cm). Not ascending further.")
                return

            direction = 1 if going_up else -1
            try:
                self.tello.send_rc_control(0, 0, direction * FINE_TUNE_SPEED, 0)
                time.sleep(FINE_TUNE_PULSE_SEC)
            except Exception:
                return
            finally:
                try:
                    self.tello.send_rc_control(0, 0, 0, 0)  # always fully stop after each pulse
                except Exception:
                    pass

            time.sleep(FINE_TUNE_SETTLE_SEC)  # let the barometer reading settle before re-measuring

        try:
            final_height = self.tello.get_height()
        except Exception:
            final_height = "?"
        print(f"[DRONE] Fine-tune stopped after {FINE_TUNE_MAX_ITERATIONS} pulses at ~{final_height} cm.")

    def _start_exact_turn(self, target_players: bool) -> bool:
        """Start an exact 180-degree turn without blocking the live video loop."""
        if self.use_webcam or not self.allow_flight:
            self.facing_players = target_players
            self._turn_error = None
            self._turn_done.set()
            direction = "players (RED)" if target_players else "away (GREEN)"
            print(f"[SIM] exact 180-degree turn -> {direction}")
            return True

        if not self.airborne:
            print("[DRONE] Turn ignored because the drone is not airborne.")
            return False

        with self._turn_lock:
            if self._turn_thread is not None and self._turn_thread.is_alive():
                return False

            # Avoid an unnecessary extra 180 degrees after a restart.
            if self.facing_players == target_players:
                self._turn_error = None
                self._turn_done.set()
                return True

            self._turn_error = None
            self._turn_target_players = target_players
            self._turn_done.clear()

            def _worker():
                try:
                    if target_players:
                        print(f"[DRONE] RED transition: rotating {ROTATE_DEGREES} degrees clockwise to players.")
                        self.tello.rotate_clockwise(ROTATE_DEGREES)
                    else:
                        print(f"[DRONE] GREEN transition: rotating {ROTATE_DEGREES} degrees counter-clockwise away.")
                        self.tello.rotate_counter_clockwise(ROTATE_DEGREES)
                    self.facing_players = target_players
                except Exception as exc:
                    self._turn_error = exc
                    print(f"[DRONE] Rotation failed: {exc}")
                finally:
                    self._turn_target_players = None
                    self._turn_done.set()

            self._turn_thread = threading.Thread(
                target=_worker,
                daemon=True,
                name="tello-180-turn",
            )
            self._turn_thread.start()
            return True

    def start_face_players(self) -> bool:
        """Asynchronously return exactly 180 degrees to the original player-facing position."""
        return self._start_exact_turn(target_players=True)

    def start_face_away(self) -> bool:
        """Asynchronously rotate exactly 180 degrees away from the players."""
        return self._start_exact_turn(target_players=False)

    def turn_finished(self) -> bool:
        return self._turn_done.is_set()

    def wait_for_turn(self, timeout: float = None) -> bool:
        self._turn_done.wait(timeout=timeout)
        return self._turn_done.is_set()

    def turn_error(self):
        return self._turn_error

    # Blocking compatibility helpers.
    def face_players(self):
        self.start_face_players()
        self.wait_for_turn()

    def face_away(self):
        self.start_face_away()
        self.wait_for_turn()

    def quick_turn(self, clockwise: bool):
        """Kept for compatibility; gameplay now uses exact 180-degree SDK rotations."""
        target_players = bool(clockwise)
        self._start_exact_turn(target_players=target_players)
        self.wait_for_turn()

    def land(self):
        """Land only when the user exits or a safety stop requires it."""
        if self.use_webcam or not self.allow_flight:
            if self.airborne:
                print("[SIM] land")
            self.airborne = False
            return

        if not self.airborne:
            return

        if self._stabilizer is not None:
            self._stabilizer.stop()
            self._stabilizer = None

        # Stop the recurring hover heartbeat before issuing the final land command.
        self._stop_hover_keepalive()

        # Do not send a landing command on top of an active 180-degree SDK command.
        self.wait_for_turn(timeout=8.0)

        try:
            self.tello.send_rc_control(0, 0, 0, 0)
        except Exception:
            pass

        try:
            print("[DRONE] Landing...")
            self.tello.land()
        finally:
            self.airborne = False

    def set_doll_eyes(self, closed: bool):
        if self.use_webcam or not self.allow_flight or not HAS_TELLO_TALENT:
            return
        try:
            self.tello.send_expansion_command("led 0 0 0" if closed else "led 255 0 0")
        except Exception:
            pass

    def close(self):
        if self.use_webcam:
            if self.cap:
                self.cap.release()
            return

        if self.tello is None:
            return

        # Normal gameplay keeps the drone airborne through results/restarts.
        # close() is reached on Q/window exit/error, so this is the final landing point.
        if self.airborne:
            self.land()

        try:
            self.tello.streamoff()
        except Exception:
            pass
        self.frame_read = None


WINDOW_NAME = "Squid Game Drone"

def _set_fullscreen():
    """Keep every gameplay screen in the same full-screen OpenCV window."""
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN,
        )
    except cv2.error:
        pass

def _draw_turning_overlay(frame, target_state: LightState, game_number: int, round_number: int):
    """Live video shown while the drone performs an exact 180-degree turn."""
    canvas = frame.copy()
    color = (0, 190, 0) if target_state == LightState.GREEN else (0, 0, 230)
    label = "TURNING 180 - PREPARING GREEN LIGHT" if target_state == LightState.GREEN else "TURNING 180 - PREPARING RED LIGHT"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 64), color, -1)
    cv2.putText(
        canvas,
        f"GAME {game_number} | ROUND {round_number} | {label}",
        (16, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def draw_hud(frame, state: LightState, eliminated_count: int, motion_ratio: float,
             remaining_sec: float = 0.0, game_number: int = 1, round_number: int = 1):
    color = (0, 190, 0) if state == LightState.GREEN else (0, 0, 230)
    light_label = "GREEN LIGHT - GO!" if state == LightState.GREEN else "RED LIGHT - FREEZE!"
    label = f"GAME {game_number}  |  ROUND {round_number}  |  {light_label}"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 58), color, -1)

    countdown = f"{max(0.0, remaining_sec):.1f}s"
    (tw, _), _ = cv2.getTextSize(countdown, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    max_label_width = max(180, frame.shape[1] - tw - 54)
    label_scale = 0.88
    while label_scale > 0.52:
        label_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, 2)[0][0]
        if label_width <= max_label_width:
            break
        label_scale -= 0.05
    cv2.putText(frame, label, (14, 39), cv2.FONT_HERSHEY_SIMPLEX, label_scale, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, countdown, (frame.shape[1] - tw - 16, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"out: {eliminated_count}", (10, frame.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    return frame


def save_result_evidence(
    frame,
    name: str,
    track_id: int,
    bbox: tuple,
    status: str,
    detector=None,
):
    """Save visual evidence for a WIN or OUT result."""

    evidence_frame = frame.copy()

    x1, y1, x2, y2 = bbox

    if status == "WIN":
        color = (0, 220, 0)
    else:
        color = (0, 0, 255)

    cv2.rectangle(
        evidence_frame,
        (x1, y1),
        (x2, y2),
        color,
        3,
    )

    cv2.putText(
        evidence_frame,
        f"{name} - {status}",
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )

    # Keep motion contours for OUT evidence
    if (
        status == "OUT"
        and detector is not None
        and detector.last_thresh is not None
    ):
        h, w = detector.last_thresh.shape

        rx1 = max(0, x1)
        ry1 = max(0, y1)
        rx2 = min(w, x2)
        ry2 = min(h, y2)

        if rx2 > rx1 and ry2 > ry1:

            region = detector.last_thresh[
                ry1:ry2,
                rx1:rx2
            ]

            contours, _ = cv2.findContours(
                region,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            for contour in contours:

                shifted = contour + np.array(
                    [[[rx1, ry1]]]
                )

                cv2.drawContours(
                    evidence_frame,
                    [shifted],
                    -1,
                    (0, 255, 255),
                    2,
                )

    cv2.putText(
        evidence_frame,
        time.strftime("%Y-%m-%d %H:%M:%S"),
        (10, evidence_frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    safe_name = (
        name
        .replace(" ", "_")
        .replace("/", "_")
    )

    filename = (
        f"{int(time.time() * 1000)}_"
        f"{safe_name}_"
        f"{status.lower()}.jpg"
    )

    filepath = os.path.join(
        EVIDENCE_DIR,
        filename,
    )

    os.makedirs(EVIDENCE_DIR, exist_ok=True)

    cv2.imwrite(
        filepath,
        evidence_frame,
    )

    return filepath


def run_lobby_phase(drone: DroneController, required_players: int,
                     tracker: "PersonTracker", registry: "PlayerRegistry",
                     face_identifier: "FaceIdentifier" = None) -> bool:
    """English documentation."""

    _set_fullscreen()

    print(f"[LOBBY] Waiting for {required_players} confirmed players (each player must stay visible for "
          f"{CONFIRM_SECONDS} seconds). Press 's' to start manually or 'q' to cancel.")

    frame_count = 0
    announced_ids = set()  
    lobby_start = time.time()

    while True:
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

        
        
        
        for tid, rec in registry.players.items():
            if tid in announced_ids:
                continue
            if rec.is_confirmed(now):
                announced_ids.add(tid)
                if rec.name != "Unknown":
                    print(f"[LOBBY] Known player detected: {rec.name}")
                else:
                    print(f"[LOBBY] New unregistered player (ID {tid}) - register later with: "
                          f"register --name \"...\" --webcam")

        frame = draw_registry(frame, registry, now)

        confirmed = registry.confirmed_count(now)
        total = len(registry.players)

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (40, 40, 40), -1)
        cv2.putText(frame, f"LOBBY - confirmed: {confirmed}/{required_players}  (total visible: {total})",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        
        
        draw_pixel_logo(frame, time.time() - lobby_start)

        
        draw_battery(frame, drone.get_battery())

        
        list_entries = [
            (rec.name if rec.name != "Unknown" else f"Player {rec.track_id}",
             COLOR_ALIVE if rec.is_confirmed(now) else COLOR_NEW)
            for rec in registry.players.values()
        ]
        draw_player_list(frame, list_entries)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False
        if key == ord("s"):
            print("[LOBBY] Manual start.")
            return True
        if confirmed >= required_players:
            print(f"[LOBBY] Confirmed players: {confirmed} - starting the game!")
            return True



def _reidentify_locked_roster(
    frame,
    raw_track_ids,
    raw_boxes,
    face_identifier,
    name_to_canonical_id,
    track_to_canonical_id,
    registry,
):
    """Re-bind NEW tracker IDs to the SAME players locked at game start.

    The roster is locked by player NAME. A new tracker ID is accepted only
    when FaceIdentifier recognizes it as one of the names that were already
    in the starting roster. Unknown people and registered people who were not
    in the starting roster are ignored.
    """
    if face_identifier is None or not raw_track_ids:
        return 0

    raw_box_map = dict(zip(raw_track_ids, raw_boxes))

    try:
        recognized = face_identifier.identify_players(frame, raw_box_map)
    except Exception as exc:
        print(f"[REID] Face re-identification warning: {exc}")
        return 0

    rebound = 0

    for raw_tid, name in recognized.items():
        canonical_tid = name_to_canonical_id.get(name)

        # IMPORTANT: only names locked when the game started are allowed.
        if canonical_tid is None:
            continue

        # Remove an older tracker-ID binding for this same player.
        for old_raw_tid, old_canonical_tid in list(track_to_canonical_id.items()):
            if old_canonical_tid == canonical_tid and old_raw_tid != raw_tid:
                del track_to_canonical_id[old_raw_tid]

        previous = track_to_canonical_id.get(raw_tid)
        track_to_canonical_id[raw_tid] = canonical_tid

        box = raw_box_map.get(raw_tid)
        if box is not None:
            # Update the CANONICAL player record, never create a gameplay
            # participant using the new raw tracker ID.
            registry.update([canonical_tid], [box])
            rec = registry.players.get(canonical_tid)
            if rec is not None:
                rec.name = name

        if previous != canonical_tid:
            print(
                f"[REID] {name}: tracker ID {raw_tid} -> "
                f"game player ID {canonical_tid}"
            )
            rebound += 1

    return rebound

def _run_round(drone: DroneController, use_webcam: bool, tracker: "PersonTracker",
               registry: "PlayerRegistry", announcer: "VoiceAnnouncer",
               face_identifier: "FaceIdentifier", cap_color_to_name: dict,
               use_evidence: bool, blindfold: bool, sack_race: bool, long_range: bool,
               hat_images: dict, hat_assignment: dict, hat_by_name: dict,
               traffic_lights: "TrafficLightController", round_number: int = 1) -> bool:
    """Run one complete game while keeping the drone airborne until the user quits."""
    _set_fullscreen()
    game_number = round_number
    detector = PerPlayerMotionChecker()

    # Normally takeoff already happened before the lobby so recognition runs while airborne.
    # This is only a fallback for a restarted session after a safety landing.
    if not drone.airborne:
        took_off = drone.takeoff()
        if not took_off:
            print("[SAFETY] Game cancelled because the drone could not take off safely.")
            return False

    safety = SafetyMonitor(use_webcam=use_webcam)
    time_mult = SACK_RACE_TIME_MULTIPLIER if sack_race else 1.0

    start_now = time.time()

    # Freeze the roster at game start:
    # only confirmed, registered players from the lobby are allowed to play.
    participants = [
        rec for rec in registry.players.values()
        if rec.is_confirmed(start_now) and rec.name != "Unknown"
    ]

    participant_order = [rec.track_id for rec in participants]
    participant_ids = set(participant_order)

    participant_names = {
        rec.track_id: rec.name
        for rec in participants
    }

    # Stable identity mapping:
    # canonical IDs are the IDs the players had at game start.
    # If the tracker changes an ID after the drone turns away/back, Face ID
    # can bind the new raw tracker ID back to the SAME canonical player.
    name_to_canonical_id = {
        participant_names[tid]: tid
        for tid in participant_order
    }
    track_to_canonical_id = {
        tid: tid
        for tid in participant_order
    }

    # Remove lobby-only unknown IDs so they do not appear once the game starts.
    for tid in list(registry.players.keys()):
        if tid not in participant_ids:
            del registry.players[tid]

    print(
        f"[PLAYERS] Roster locked with {len(participant_ids)} registered player(s). "
        "Only these names can be re-identified during gameplay."
    )

    if not participant_ids:
        print("[WARN] No players are available for this game.")
        return False

    print(f"[INFO] Game {game_number} starting with {len(participant_ids)} player(s).")

    result_by_id = {}
    arrival_order = []
    out_order = []
    eliminated_ids = set()
    last_edge_warning = {}
    moving_streak = {}

    # Initial position is assumed to face the players. Before GREEN, turn exactly 180 degrees away.
    state = LightState.GREEN
    cycle_number = 1
    transition_target = LightState.GREEN
    state_deadline = float("inf")
    red_transition_start = 0.0
    eye_screen_until = 0.0
    red_motion_enable_at = float("inf")
    frame_count = 0
    last_frame = None
    game_start = time.time()
    user_quit = False
    safety_stop = False
    last_motion_check_at = 0.0

    # Re-identification is only useful while the drone is facing the players.
    # Throttle it so CPU Face Recognition does not run every video frame.
    next_reid_at = 0.0
    reid_interval_sec = 0.45
    # Force exactly one identity refresh after every completed return to RED.
    red_turn_reid_done = True

    detector.reset()
    traffic_lights.off()
    drone.set_doll_eyes(closed=True)
    drone.start_face_away()

    print("[INFO] Press Q to finish the session and land. The drone stays airborne through results and Restart.")

    while True:
        frame = drone.get_frame()
        if not safety.note_frame(frame):
            safety_stop = True
            break
        if frame is None:
            continue
        frame_count += 1

        if not safety.check_ongoing(drone):
            safety_stop = True
            break

        frame_h, frame_w = frame.shape[:2]
        track_ids, boxes = tracker.process(frame)

        # Convert raw tracker IDs into the stable/canonical IDs that were
        # locked when the game started. New raw IDs are NOT players yet.
        mapped_pairs = []
        visible_canonical_ids = set()

        for raw_tid, box in zip(track_ids, boxes):
            canonical_tid = track_to_canonical_id.get(raw_tid)
            if canonical_tid in participant_ids:
                mapped_pairs.append((canonical_tid, box))
                visible_canonical_ids.add(canonical_tid)

        mapped_track_ids = [tid for tid, _ in mapped_pairs]
        mapped_boxes = [box for _, box in mapped_pairs]
        now = registry.update(mapped_track_ids, mapped_boxes)

        # Keep the canonical records named even if PlayerRegistry had to
        # recreate a stale record after the drone was facing away.
        for canonical_tid in visible_canonical_ids:
            rec = registry.players.get(canonical_tid)
            if rec is not None:
                rec.name = participant_names[canonical_tid]

        # When the drone has returned to face the players, re-run Face ID
        # only as needed. This lets a new tracker ID re-bind to Rahaf/Jood/etc.
        # It NEVER adds a new name to the game's roster.
        unresolved_for_reid = participant_ids.difference(result_by_id.keys())
        missing_visible = unresolved_for_reid.difference(visible_canonical_ids)
        facing_players_now = drone.facing_players and drone.turn_finished()

        force_red_reid = (
            transition_target == LightState.RED
            and drone.turn_finished()
            and not red_turn_reid_done
        )

        if (
            face_identifier is not None
            and facing_players_now
            and (missing_visible or force_red_reid)
            and time.time() >= next_reid_at
        ):
            _reidentify_locked_roster(
                frame,
                track_ids,
                boxes,
                face_identifier,
                name_to_canonical_id,
                track_to_canonical_id,
                registry,
            )
            if force_red_reid:
                red_turn_reid_done = True
            next_reid_at = time.time() + reid_interval_sec

            # Rebuild visible mappings immediately after a successful re-ID
            # so RED motion detection can use the canonical player this frame.
            mapped_pairs = []
            visible_canonical_ids = set()
            for raw_tid, box in zip(track_ids, boxes):
                canonical_tid = track_to_canonical_id.get(raw_tid)
                if canonical_tid in participant_ids:
                    mapped_pairs.append((canonical_tid, box))
                    visible_canonical_ids.add(canonical_tid)

            if mapped_pairs:
                registry.update(
                    [tid for tid, _ in mapped_pairs],
                    [box for _, box in mapped_pairs],
                )
                for canonical_tid in visible_canonical_ids:
                    rec = registry.players.get(canonical_tid)
                    if rec is not None:
                        rec.name = participant_names[canonical_tid]

        # Long-range cap-color fallback remains restricted to the locked roster.
        if long_range and cap_color_to_name and frame_count % FACE_ID_INTERVAL_FRAMES == 0:
            for tid in participant_ids:
                rec = registry.players.get(tid)
                if rec is None or rec.name != "Unknown" or rec.bbox is None:
                    continue
                color = detect_cap_color(frame, rec.bbox)
                if color and color in cap_color_to_name:
                    rec.name = cap_color_to_name[color]
                    participant_names[tid] = rec.name

        for tid in participant_ids:
            rec = registry.players.get(tid)
            if rec is not None and rec.name != "Unknown":
                participant_names[tid] = rec.name

        # ------------------------------------------------------------
        # 180-degree transition state. The video loop never blocks here.
        # ------------------------------------------------------------
        if transition_target is not None:
            turn_done = drone.turn_finished()

            if transition_target == LightState.RED:
                # Preserve the requested full-screen eye warning. If the physical turn takes
                # longer than 3 seconds, keep the warning up until the drone is actually facing players.
                warning_done = now >= eye_screen_until
                if not (turn_done and warning_done):
                    opening_span = min(0.75, max(0.1, RED_EYE_FULLSCREEN_SEC))
                    eye_openness = min(1.0, max(0.0, (now - red_transition_start) / opening_span))
                    display_frame = draw_fullscreen_eyes(
                        frame,
                        eye_openness,
                        max(0.0, eye_screen_until - now),
                        game_number=game_number,
                        round_number=cycle_number,
                    )
                    cv2.imshow(WINDOW_NAME, display_frame)
                    last_frame = display_frame
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        user_quit = True
                        break
                    continue

                if drone.turn_error() is not None:
                    print("[SAFETY] RED turn failed. Ending the session.")
                    safety_stop = True
                    break

                transition_target = None
                state = LightState.RED
                traffic_lights.red()
                detector.reset()
                moving_streak.clear()
                red_motion_enable_at = time.time() + RED_LIGHT_GRACE_PERIOD_SEC
                state_deadline = time.time() + random.uniform(RED_LIGHT_MIN_SEC, RED_LIGHT_MAX_SEC) * time_mult
                play_beep(700, 250)
                print("[LIGHT] RED active: drone is facing players; motion detection is synchronized.")

            else:
                if not turn_done:
                    display_frame = _draw_turning_overlay(
                        frame,
                        LightState.GREEN,
                        game_number,
                        cycle_number,
                    )
                    cv2.imshow(WINDOW_NAME, display_frame)
                    last_frame = display_frame
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        user_quit = True
                        break
                    continue

                if drone.turn_error() is not None:
                    print("[SAFETY] GREEN turn failed. Ending the session.")
                    safety_stop = True
                    break

                transition_target = None
                state = LightState.GREEN
                traffic_lights.green()
                drone.set_doll_eyes(closed=True)
                announcer.say_state(VOICE_LANG_GREEN)
                state_deadline = time.time() + random.uniform(GREEN_LIGHT_MIN_SEC, GREEN_LIGHT_MAX_SEC) * time_mult
                detector.reset()
                moving_streak.clear()
                print("[LIGHT] GREEN active: 180-degree turn complete; timer starts now.")

        # ------------------------------------------------------------
        # Start the next physical turn only after the active light timer ends.
        # The next light timer never includes the rotation time.
        # ------------------------------------------------------------
        if transition_target is None and now >= state_deadline:
            if state == LightState.GREEN:
                # Transition period: no color is active until the drone is
                # physically back facing the players.
                traffic_lights.off()
                announcer.say_state(VOICE_LANG_RED)
                drone.set_doll_eyes(closed=False)
                red_transition_start = time.time()
                eye_screen_until = red_transition_start + RED_EYE_FULLSCREEN_SEC
                transition_target = LightState.RED
                red_turn_reid_done = False
                state_deadline = float("inf")
                detector.reset()
                moving_streak.clear()
                drone.start_face_players()
                continue

            cycle_number += 1
            # Transition period: switch both colors off while the drone turns away.
            traffic_lights.off()
            transition_target = LightState.GREEN
            state_deadline = float("inf")
            detector.reset()
            moving_streak.clear()
            drone.set_doll_eyes(closed=True)
            drone.start_face_away()
            continue

        unresolved_ids = participant_ids.difference(result_by_id.keys())

        # A player can finish only while GREEN is fully active (never during a turn).
        if transition_target is None and state == LightState.GREEN:
            if sack_race:
                finish_y = int(frame_h * FINISH_LINE_Y_RATIO)
                cv2.line(frame, (0, finish_y), (frame_w, finish_y), (255, 0, 255), 2)
                cv2.putText(frame, "FINISH LINE", (10, finish_y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                finish_candidates = []
                for tid in unresolved_ids:
                    rec = registry.players.get(tid)
                    if rec is None or rec.bbox is None:
                        continue
                    _, y1, _, _ = rec.bbox
                    if y1 <= finish_y:
                        finish_candidates.append((y1, participant_order.index(tid), tid))
                for _, _, tid in sorted(finish_candidates):
                    arrival_order.append(tid)
                    rank = len(arrival_order)
                    rec = registry.players.get(tid)
                    evidence_path = None
                    if use_evidence and rec is not None and rec.bbox is not None:
                        evidence_path = save_result_evidence(
                            frame, participant_names[tid], tid, rec.bbox, "WIN"
                        )
                        print(f"[EVIDENCE] WIN saved: {evidence_path}")
                    result_by_id[tid] = {
                        "name": participant_names[tid], "status": "WIN",
                        "round": cycle_number, "rank": rank,
                        "evidence": evidence_path,
                    }
                    print(f"[FINISH] #{rank} {participant_names[tid]} won in round {cycle_number}.")
                    announcer.say(f"{participant_names[tid]} finished in position {rank}!")
            elif now - game_start >= PROXIMITY_WIN_GRACE_SEC:
                finish_candidates = []
                for tid in unresolved_ids:
                    rec = registry.players.get(tid)
                    if rec is None or rec.bbox is None:
                        continue
                    _, y1, _, y2 = rec.bbox
                    proximity = (y2 - y1) / frame_h
                    if proximity >= PROXIMITY_WIN_RATIO:
                        finish_candidates.append((-proximity, participant_order.index(tid), tid))
                for _, _, tid in sorted(finish_candidates):
                    arrival_order.append(tid)
                    rank = len(arrival_order)
                    rec = registry.players.get(tid)
                    evidence_path = None
                    if use_evidence and rec is not None and rec.bbox is not None:
                        evidence_path = save_result_evidence(
                            frame, participant_names[tid], tid, rec.bbox, "WIN"
                        )
                        print(f"[EVIDENCE] WIN saved: {evidence_path}")
                    result_by_id[tid] = {
                        "name": participant_names[tid], "status": "WIN",
                        "round": cycle_number, "rank": rank,
                        "evidence": evidence_path,
                    }
                    print(f"[FINISH] #{rank} {participant_names[tid]} won in round {cycle_number}.")
                    announcer.say(f"{participant_names[tid]} finished in position {rank}!")

        unresolved_ids = participant_ids.difference(result_by_id.keys())

        if blindfold:
            margin = frame_w * EDGE_WARNING_MARGIN_RATIO
            for tid in unresolved_ids:
                rec = registry.players.get(tid)
                if rec is None or rec.bbox is None:
                    continue
                x1, _, x2, _ = rec.bbox
                if x1 < margin or x2 > frame_w - margin:
                    last = last_edge_warning.get(tid, 0)
                    if now - last >= EDGE_WARNING_COOLDOWN_SEC:
                        announcer.say("Careful, you're near the edge!")
                        last_edge_warning[tid] = now

        # Motion detection begins only when RED is active, the 180 turn is complete,
        # and the short post-turn grace period has elapsed.
        if (
            transition_target is None
            and state == LightState.RED
            and now >= red_motion_enable_at
            and unresolved_ids
            and now - last_motion_check_at >= MOTION_SAMPLE_INTERVAL_SEC
        ):
            last_motion_check_at = now
            box_map = {
                tid: registry.players[tid].bbox
                for tid in unresolved_ids
                if tid in registry.players and registry.players[tid].bbox is not None
            }
            moving_ids = detector.check(frame, box_map)

            for tid in list(moving_streak):
                if tid not in moving_ids:
                    moving_streak[tid] = 0

            for tid in moving_ids:
                if tid not in unresolved_ids:
                    continue
                moving_streak[tid] = moving_streak.get(tid, 0) + 1
                if moving_streak[tid] < MOTION_CONFIRM_FRAMES:
                    continue

                eliminated_ids.add(tid)
                out_order.append(tid)
                rec = registry.players.get(tid)
                evidence_path = None
                if use_evidence and rec is not None and rec.bbox is not None:
                    evidence_path = save_result_evidence(
                        frame, participant_names[tid], tid, rec.bbox, "OUT", detector
                    )
                    print(f"[EVIDENCE] OUT saved: {evidence_path}")
                result_by_id[tid] = {
                    "name": participant_names[tid], "status": "OUT",
                    "round": cycle_number, "rank": None,
                    "evidence": evidence_path,
                }
                print(f"[OUT] {participant_names[tid]} moved during RED LIGHT in round {cycle_number}.")
                announcer.say(f"{participant_names[tid]} is out!")

        display_frame = frame.copy()
        display_frame = draw_registry(display_frame, registry, now, eliminated_ids)
        if hat_images:
            display_frame = draw_hats(
                display_frame, registry, hat_images, hat_assignment, eliminated_ids,
                hat_identity_assignment=hat_by_name, auto_assign=False,
            )

        display_frame = draw_hud(
            display_frame, state, len(eliminated_ids), 0.0,
            max(0.0, state_deadline - now) if state_deadline != float("inf") else 0.0,
            game_number=game_number, round_number=cycle_number,
        )

        rank_colors = {1: (0, 220, 255), 2: (192, 192, 192), 3: (50, 127, 205)}
        live_entries = []
        for tid in participant_order:
            result = result_by_id.get(tid)
            name = participant_names.get(tid, f"Player {tid}")
            if result is None:
                live_entries.append((name, COLOR_ALIVE))
            elif result["status"] == "WIN":
                rank = result["rank"]
                live_entries.append((f"{rank}. {name} - WIN R{result['round']}", rank_colors.get(rank, COLOR_ALIVE)))
            else:
                live_entries.append((f"{name} - OUT R{result['round']}", COLOR_OUT))
        draw_player_list(display_frame, live_entries)

        cv2.imshow(WINDOW_NAME, display_frame)
        last_frame = display_frame

        if participant_ids.issubset(result_by_id.keys()):
            print(f"[INFO] Game {game_number} complete: every player is WIN or OUT.")
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            user_quit = True
            break

    # No traffic light remains on after a round, on Q, or on safety stop.
    traffic_lights.off()

    # Only a genuine safety stop can force a landing without Q.
    # Normal game completion and the results screen keep the drone in the air.
    if safety_stop and drone.airborne:
        print("[SAFETY] Emergency/safety landing override.")
        drone.land()

    if user_quit:
        print("[INFO] Q pressed. Ending the session; cleanup will land the drone.")
        return False

    # If a safety stop happened, mark unresolved players OUT in the current round.
    for tid in participant_ids:
        if tid not in result_by_id:
            eliminated_ids.add(tid)
            out_order.append(tid)
            rec = registry.players.get(tid)
            evidence_path = None
            if use_evidence and last_frame is not None and rec is not None and rec.bbox is not None:
                evidence_path = save_result_evidence(
                    last_frame, participant_names[tid], tid, rec.bbox, "OUT"
                )
                print(f"[EVIDENCE] OUT saved: {evidence_path}")
            result_by_id[tid] = {
                "name": participant_names[tid], "status": "OUT",
                "round": cycle_number, "rank": None,
                "evidence": evidence_path,
            }

    winner_rows = [result_by_id[tid] for tid in arrival_order if tid in result_by_id]
    out_rows = [result_by_id[tid] for tid in out_order if tid in result_by_id]
    final_rows = winner_rows + out_rows
    winners = [row["name"] for row in winner_rows]

    print("=" * 58)
    print(f"[RESULT] GAME {game_number}")
    for row in final_rows:
        if row["status"] == "WIN":
            print(f"  #{row['rank']} {row['name']} - WIN - round {row['round']}")
        else:
            print(f"  OUT {row['name']} - round {row['round']}")
    print("=" * 58)

    if winners:
        update_leaderboard(winners)
        announcer.say(f"Game {game_number} complete. First place is {winners[0]}!")
    else:
        announcer.say(f"Game {game_number} complete. No winners.")

    base_frame = last_frame if last_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
    results_frame = draw_results_dashboard(base_frame, final_rows, game_number)
    _set_fullscreen()
    cv2.imshow(WINDOW_NAME, results_frame)
    return _wait_for_next_round(results_frame, game_number)

def _wait_for_next_round(base_frame, round_number: int) -> bool:
    """Wait on the final results screen for Restart or Quit."""

    window_name = WINDOW_NAME
    _set_fullscreen()

    h, w = base_frame.shape[:2]

    button_w = min(
        190,
        max(
            120,
            int(w * 0.34)
        )
    )

    button_h = 44
    gap = 18

    total_w = (
        button_w * 2
        + gap
    )

    start_x = (
        w - total_w
    ) // 2

    button_y = h - 62

    restart_rect = (
        start_x,
        button_y,
        start_x + button_w,
        button_y + button_h
    )

    quit_rect = (
        start_x + button_w + gap,
        button_y,
        start_x + button_w + gap + button_w,
        button_y + button_h
    )

    action = {
        "value": None
    }

    def inside(rect, x, y):

        x1, y1, x2, y2 = rect

        return (
            x1 <= x <= x2
            and
            y1 <= y <= y2
        )

    def mouse_callback(
        event,
        x,
        y,
        flags,
        param
    ):

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if inside(
            restart_rect,
            x,
            y
        ):
            action["value"] = "restart"

        elif inside(
            quit_rect,
            x,
            y
        ):
            action["value"] = "quit"

    cv2.imshow(
        window_name,
        base_frame
    )

    cv2.setMouseCallback(
        window_name,
        mouse_callback
    )

    print(
        "[RESULTS] Click RESTART or QUIT."
    )

    print(
        "[RESULTS] Keyboard shortcuts: R = Restart, Q = Quit."
    )

    try:

        while True:

            cv2.imshow(
                window_name,
                base_frame
            )

            key = (
                cv2.waitKey(
                    GAME_OVER_POLL_MS
                )
                & 0xFF
            )

            # =====================
            # Restart
            # =====================

            if (
                action["value"]
                == "restart"
                or
                key == ord("r")
            ):

                print(
                    "[GAME] Restart selected."
                )

                return True

            # =====================
            # Quit
            # =====================

            if (
                action["value"]
                == "quit"
                or
                key == ord("q")
                or
                key == 27
            ):

                print(
                    "[GAME] Quit selected."
                )

                return False

            # Window closed manually
            try:

                visible = cv2.getWindowProperty(
                    window_name,
                    cv2.WND_PROP_VISIBLE
                )

                if visible < 1:
                    return False

            except cv2.error:
                return False

    finally:

        try:
            cv2.setMouseCallback(
                window_name,
                lambda *args: None
            )

        except Exception:
            pass


def cmd_play(use_webcam: bool, allow_flight: bool, skip_lobby: bool = False,
             use_face_id: bool = True, use_voice: bool = True,
             use_evidence: bool = True, modes: list = None):
    modes = modes or ["classic"]
    blindfold = "blindfold" in modes
    sack_race = "sack-race" in modes
    long_range = "long-range" in modes

    if modes != ["classic"]:
        print(f"[INFO] Active modes: {', '.join(modes)}")

    drone = DroneController(use_webcam=use_webcam, allow_flight=allow_flight)
    announcer = VoiceAnnouncer(enabled=use_voice)

    # ESP32 is connected by USB Serial. If it is not connected, the game
    # still works normally; only the external lights are disabled.
    traffic_lights = TrafficLightController()
    traffic_lights.off()

    tracker = None

    try:
        # Load the heavy models before takeoff so battery is not wasted during model startup.
        # Recognition itself starts only after the drone is airborne in the lobby.
        tracker = PersonTracker()
        registry = PlayerRegistry()

        hat_images = load_hat_images() if HAT_OVERLAY_ENABLED else {}
        hat_assignment = {}
        hat_by_name = {}

        face_identifier = None
        if use_face_id:
            try:
                face_identifier = FaceIdentifier()
                if not face_identifier.known:
                    print(
                        "[INFO] No registered players found in players.json. "
                        "Track IDs will be shown until you register players with: "
                        "register --name <name> --webcam"
                    )
            except ImportError:
                print("[WARN] insightface is not installed. Continuing with track IDs only.")
                print("       Install with: py -m pip install insightface onnxruntime")
                face_identifier = None

        cap_color_to_name = {}
        if long_range:
            cap_colors = load_cap_colors()
            cap_color_to_name = {color: name for name, color in cap_colors.items()}
            if not cap_color_to_name:
                print(
                    "[INFO] long-range mode is enabled, but no cap colors are registered. "
                    'Example: register --name "..." --webcam --cap-color red'
                )

        # IMPORTANT: take off BEFORE lobby recognition. The drone now recognizes players while hovering.
        if not drone.takeoff():
            print("[SAFETY] Session cancelled because takeoff failed.")
            return

        print("[DRONE] Keep the drone initially facing the players. Recognition is now starting while airborne.")
        _set_fullscreen()

        if not skip_lobby:
            started = run_lobby_phase(
                drone,
                REQUIRED_CONFIRMED_PLAYERS,
                tracker,
                registry,
                face_identifier,
            )
            if not started:
                print("[INFO] Lobby cancelled. Q/exit will land the drone now.")
                return

            if hat_images:
                hats_ready = run_hat_selection_phase(
                    drone,
                    tracker,
                    registry,
                    hat_images,
                    face_identifier=face_identifier,
                    hat_assignment=hat_assignment,
                    hat_by_name=hat_by_name,
                )
                if not hats_ready:
                    print("[INFO] Hat selection cancelled. Q/exit will land the drone now.")
                    return
                _set_fullscreen()
        elif hat_images:
            print("[HATS] --skip-lobby also skips hat selection.")

        keep_playing = True
        round_number = 1
        while keep_playing:
            keep_playing = _run_round(
                drone,
                use_webcam,
                tracker,
                registry,
                announcer,
                face_identifier,
                cap_color_to_name,
                use_evidence,
                blindfold,
                sack_race,
                long_range,
                hat_images,
                hat_assignment,
                hat_by_name,
                traffic_lights,
                round_number=round_number,
            )
            if keep_playing:
                round_number += 1
                _set_fullscreen()

    finally:
        # This is reached after Q, window exit, cancellation, or an unexpected error.
        # Normal game completion/restart does not reach this point, so the drone keeps hovering.
        traffic_lights.close()
        if tracker is not None:
            tracker.close()
        drone.close()
        cv2.destroyAllWindows()
        announcer.close()
