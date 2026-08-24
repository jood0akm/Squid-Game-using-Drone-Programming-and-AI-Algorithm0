"""Central configuration for the Squid Game Drone project."""

import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_FRAMES_DIR = os.path.join(DATA_DIR, "raw")
LABELS_DIR = os.path.join(DATA_DIR, "labels")
MOTION_LOG_CSV = os.path.join(DATA_DIR, "motion_log.csv")

# Capture / motion baseline
CAPTURE_FPS = 5
MOTION_AREA_RATIO_THRESHOLD = 0.08
BLUR_KERNEL = (21, 21)
MOTION_CONFIRM_FRAMES = 5
# Conservative motion filtering to avoid false eliminations from camera noise or tracker jitter.
MOTION_PIXEL_DIFF_THRESHOLD = 45
MOTION_MEAN_DIFF_THRESHOLD = 8.0
MOTION_CENTER_SHIFT_THRESHOLD = 0.065
MOTION_SIZE_CHANGE_THRESHOLD = 0.12
MOTION_ROI_INSET_X = 0.14
MOTION_ROI_INSET_Y = 0.08
MOTION_REFERENCE_BLEND = 0.04
MOTION_SAMPLE_INTERVAL_SEC = 0.05

# Game timing
GREEN_LIGHT_MIN_SEC = 2.0
GREEN_LIGHT_MAX_SEC = 5.0
# Red light is intentionally a little longer than before.
RED_LIGHT_MIN_SEC = 10
RED_LIGHT_MAX_SEC = 15
RED_LIGHT_GRACE_PERIOD_SEC = 0.85


# Win-by-approach settings
PROXIMITY_WIN_RATIO = 0.62
PROXIMITY_WIN_GRACE_SEC = 5.0

# Flight / doll turn
ROTATE_DEGREES = 180
QUICK_TURN_ENABLED = True
QUICK_TURN_YAW_SPEED = 60
QUICK_TURN_BURST_SEC = 0.70
HAS_TELLO_TALENT = False

# Target hover height after takeoff (roughly human eye level). Applied with a
# SINGLE discrete move_up() SDK command right after takeoff — not a
# continuous correction loop — then the Tello's own internal barometer/IMU/
# optical-flow hold keeps it there, same as it does at any other height.
# ⚠️ SAFETY: verify your ceiling clearance above this height (plus a safety
# margin — light fixtures, fans, low ceilings) BEFORE flying. The Tello has
# no upward obstacle sensor.
TARGET_HOVER_HEIGHT_ENABLED = True
TARGET_HOVER_HEIGHT_CM = 165
MOVE_COMMAND_MIN_CM = 20   # Tello SDK's move_up/move_down minimum valid distance

# Fine-tuning pass AFTER the coarse move_up() above, to close the remaining
# gap below the SDK's 20cm move-command floor. Deliberately bounded and
# defensive after the earlier ceiling-strike incident:
#   - short pulses, full stop between each one, real height re-measured
#     before deciding whether to continue
#   - a hard height ceiling that can NEVER be crossed by an upward pulse,
#     independent of everything else
#   - a hard cap on the number of pulses (never an open-ended loop)
FINE_TUNE_HEIGHT_ENABLED = True
FINE_TUNE_TOLERANCE_CM = 0          # no margin — chase exact target (barometer noise + the
                                     # FINE_TUNE_MAX_ITERATIONS cap below still guarantee it stops)
FINE_TUNE_MAX_ITERATIONS = 5        # absolute cap on correction pulses
FINE_TUNE_PULSE_SEC = 0.15          # each pulse is this short, then a full stop
FINE_TUNE_SPEED = 10                # low vertical speed during a pulse (0-100 scale)
FINE_TUNE_SETTLE_SEC = 0.3          # pause after each pulse before re-measuring height
FINE_TUNE_HARD_CEILING_CM = TARGET_HOVER_HEIGHT_CM + 25  # NEVER ascend at/above this, no matter what


# Hover stabilization: HORIZONTAL drift correction only (see HoverStabilizer
# docstring in game_engine.py). An earlier version also auto-corrected
# altitude using the barometer; that caused a real ceiling strike during
# testing (the Tello has no upward obstacle sensor), so the vertical-control
# code was removed entirely, not just disabled. Gains below are conservative
# starting points, not validated on physical hardware — enable with caution,
# in an open space, with a spotter.
HOVER_STABILIZER_ENABLED = False   # disabled by default — opt in only after reading the note above
HOVER_STABILIZER_HZ = 5.0            # correction loop rate
HOVER_SETTLE_SEC = 2.0               # ignore drift right after takeoff while it settles
HOVER_SPEED_DEADBAND_CMS = 6         # IMU/optical-flow-derived vgx/vgy: ignore drift smaller than this
HOVER_SPEED_GAIN = 0.6               # measured drift speed -> corrective rc speed
HOVER_SPEED_MAX_CORRECTION = 15      # clamp horizontal correction speed

# Video-link recovery: the Tello's Wi-Fi video feed can freeze/stall. Detect a
# frame that stops changing for too long and restart the stream automatically,
# instead of silently tracking a frozen image (which looks like "nothing is
# ever detected" in-game).
VIDEO_FREEZE_CHECK_ENABLED = True
VIDEO_FREEZE_DIFF_THRESHOLD = 1.5    # mean abs pixel diff below this counts as "unchanged"
VIDEO_FREEZE_SECONDS = 3.0           # how long a frozen feed must persist before recovery
VIDEO_FREEZE_SAMPLE_INTERVAL_SEC = 0.3

# Person tracking performance
PERSON_ASYNC_TRACKING = True
PERSON_ASYNC_MIN_INTERVAL_SEC = 0.04
PERSON_MODEL = "yolov8n.pt"
DETECTION_CONFIDENCE = 0.45
PERSON_INFERENCE_SIZE = 256
PERSON_TRACK_INTERVAL_FRAMES = 1
CONFIRM_SECONDS = 3.0
TRACK_LOST_TIMEOUT = 3.0
REQUIRED_CONFIRMED_PLAYERS = 4
USE_REID_TRACKER = False
PERSON_CLASS_ID = 0

# Face ID
PLAYERS_FILE = os.path.join(PROJECT_ROOT, "players.json")
FACE_MODEL_NAME = "buffalo_l"
FACE_CTX_ID = -1
FACE_DET_SIZE = (256, 256)
FACE_MATCH_THRESHOLD = 0.45
FACE_ID_INTERVAL_FRAMES = 60
REGISTER_TARGET_COUNT = 15

# Webcam low-latency settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 360
CAMERA_FPS = 30
CAMERA_FOURCC = "MJPG"

# Voice announcements
VOICE_ENABLED_DEFAULT = True
VOICE_RATE = 170
VOICE_LANG_GREEN = "Green light! Go!"
VOICE_LANG_RED = "Red light! Freeze!"

# Leaderboard
LEADERBOARD_FILE = os.path.join(PROJECT_ROOT, "leaderboard.json")

# Optional game modes
GAME_MODE_CHOICES = ["classic", "blindfold", "sack-race", "long-range"]
EDGE_WARNING_MARGIN_RATIO = 0.08
EDGE_WARNING_COOLDOWN_SEC = 4.0
SACK_RACE_TIME_MULTIPLIER = 1.8
FINISH_LINE_Y_RATIO = 0.15

# Long-range cap color fallback
CAP_COLOR_FILE = os.path.join(PROJECT_ROOT, "cap_colors.json")
HEAD_REGION_RATIO = 0.35
CAP_COLOR_MIN_RATIO = 0.25
CAP_COLOR_HSV_RANGES = {
    "red":    [((0, 100, 80), (10, 255, 255)), ((170, 100, 80), (180, 255, 255))],
    "blue":   [((100, 100, 60), (130, 255, 255))],
    "yellow": [((22, 100, 100), (35, 255, 255))],
    "green":  [((45, 80, 60), (85, 255, 255))],
    "orange": [((11, 120, 100), (21, 255, 255))],
    "purple": [((130, 60, 60), (160, 255, 255))],
}

# AR hats
HATS_DIR = os.path.join(PROJECT_ROOT, "assets", "hats")
HAT_OVERLAY_ENABLED = True
HAT_WIDTH_RATIO = 0.72
HAT_VERTICAL_OVERLAP_RATIO = 0.35

# Elimination evidence
EVIDENCE_DIR = os.path.join(PROJECT_ROOT, "evidence")

# Dashboard
RED_EYE_FULLSCREEN_SEC = 3.0
TITLE_TEXT = "SQUID GAME"
TITLE_POP_IN_SEC = 0.5
TITLE_HOLD_SEC = 2.0
TITLE_FADE_OUT_SEC = 0.7
TITLE_Y_OFFSET = 105
HUD_MARGIN = 15
PLAYER_LIST_FONT_SCALE = 0.62
BATTERY_LOW_THRESHOLD = 20
BATTERY_MED_THRESHOLD = 50
RESULTS_DISPLAY_SEC = 4.0
GAME_OVER_POLL_MS = 30
AUTO_NEXT_ROUND = False
NEXT_ROUND_DELAY_SEC = 4.0

for _directory in (DATA_DIR, RAW_FRAMES_DIR, LABELS_DIR, EVIDENCE_DIR, HATS_DIR):
    os.makedirs(_directory, exist_ok=True)
