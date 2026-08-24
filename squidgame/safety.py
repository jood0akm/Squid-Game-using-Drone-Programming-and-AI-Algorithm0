"""Drone safety checks and in-game safety monitoring."""

import time

SAFETY_MIN_BATTERY_TAKEOFF = 5
SAFETY_MIN_BATTERY_CONTINUE = 5
SAFETY_MAX_FLIGHT_SECONDS = 600
SAFETY_MAX_MISSED_FRAMES = 30
SAFETY_BATTERY_CHECK_INTERVAL_SEC = 15
SAFETY_MAX_TEMPERATURE_C = 90


def cmd_check(use_webcam: bool = False):
    """Connect without taking off and print a basic safety report."""
    if use_webcam:
        print("[INFO] Webcam mode does not require a drone pre-flight check.")
        return

    from djitellopy import Tello

    print("[CHECK] Connecting to drone...")
    tello = Tello()
    tello.connect()

    battery = tello.get_battery()
    temp = tello.get_temperature()
    height = tello.get_height()
    tof = None
    try:
        tof = tello.get_distance_tof()
    except Exception:
        pass

    print("=" * 44)
    print("PRE-FLIGHT SAFETY REPORT")
    print("=" * 44)
    print(f"Battery:             {battery}%")
    print(f"Temperature:         {temp} C")
    print(f"Current height:      {height} cm")
    if tof is not None:
        print(f"TOF distance sensor: {tof} cm")

    issues = []
    if battery < SAFETY_MIN_BATTERY_TAKEOFF:
        issues.append(f"Battery is below {SAFETY_MIN_BATTERY_TAKEOFF}%. Charge before takeoff.")
    if temp > SAFETY_MAX_TEMPERATURE_C:
        issues.append(f"Drone temperature is above {SAFETY_MAX_TEMPERATURE_C} C. Let it cool down.")

    print("-" * 44)
    if issues:
        print("WARNING: Not safe to fly right now:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("Drone battery and temperature are within the configured safety limits.")
        print("Reminder: standard Tello does not provide forward obstacle avoidance.")
    print("=" * 44)
    tello.end()


def cmd_fly_test(hover_seconds: float = 10.0):
    """Run a simple takeoff, hover, and landing test."""
    from djitellopy import Tello

    print("[FLY-TEST] Connecting to drone...")
    tello = Tello()
    tello.connect()

    battery = tello.get_battery()
    print(f"[FLY-TEST] Battery: {battery}%")
    if battery < SAFETY_MIN_BATTERY_TAKEOFF:
        print(f"[FLY-TEST] Battery is below {SAFETY_MIN_BATTERY_TAKEOFF}%. Test cancelled.")
        tello.end()
        return

    try:
        print("[FLY-TEST] Taking off in 3 seconds. Keep clear of the drone.")
        time.sleep(3)
        tello.takeoff()
        print(f"[FLY-TEST] Takeoff successful. Hovering for {hover_seconds:.0f} seconds. Press Ctrl+C to land.")

        start = time.time()
        while time.time() - start < hover_seconds:
            remaining = int(hover_seconds - (time.time() - start))
            print(f"[FLY-TEST] Hovering... {remaining} seconds remaining   ", end="\r")
            time.sleep(1)
        print()
    except KeyboardInterrupt:
        print("\n[FLY-TEST] Manual stop. Landing now.")
    except Exception as exc:
        print(f"[FLY-TEST] Flight test failed: {exc}")
    finally:
        if tello.is_flying:
            print("[FLY-TEST] Landing...")
            tello.land()
            print("[FLY-TEST] Landed safely.")
        else:
            print("[FLY-TEST] Drone is not airborne; no landing command is needed.")
        print("[FLY-TEST] Test finished.")
        tello.end()


class SafetyMonitor:
    """Monitor battery, video connectivity, and maximum flight duration."""

    def __init__(self, use_webcam: bool):
        self.use_webcam = use_webcam
        self.flight_start = time.time()
        self.last_battery_check = 0.0
        self.missed_frames = 0

    def note_frame(self, frame) -> bool:
        if self.use_webcam:
            return True

        if frame is None:
            self.missed_frames += 1
            if self.missed_frames >= SAFETY_MAX_MISSED_FRAMES:
                print("[SAFETY] Video connection was lost for too long. Landing immediately.")
                return False
        else:
            self.missed_frames = 0
        return True

    def check_ongoing(self, drone) -> bool:
        if self.use_webcam:
            return True

        now = time.time()
        if now - self.flight_start > SAFETY_MAX_FLIGHT_SECONDS:
            print(f"[SAFETY] Maximum flight time ({SAFETY_MAX_FLIGHT_SECONDS}s) reached. Landing.")
            return False

        if now - self.last_battery_check >= SAFETY_BATTERY_CHECK_INTERVAL_SEC:
            self.last_battery_check = now
            try:
                battery = drone.tello.get_battery()
            except Exception:
                return True
            print(f"[SAFETY] Battery: {battery}%")
            if battery < SAFETY_MIN_BATTERY_CONTINUE:
                print(f"[SAFETY] Battery below {SAFETY_MIN_BATTERY_CONTINUE}%. Landing immediately.")
                return False
        return True
