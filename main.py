"""Command-line entry point for Squid Game Drone."""

import argparse
import platform
import sys

if platform.system() == "Windows":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from squidgame.config import MOTION_AREA_RATIO_THRESHOLD, CAP_COLOR_HSV_RANGES, GAME_MODE_CHOICES
from squidgame.motion_baseline import cmd_collect, cmd_label, cmd_evaluate, cmd_sweep, cmd_motion_live
from squidgame.person_tracking import cmd_registry_test
from squidgame.hat_overlay import cmd_hats_test
from squidgame.face_id import cmd_register, cmd_list_players
from squidgame.leaderboard import cmd_leaderboard
from squidgame.safety import cmd_check, cmd_fly_test
from squidgame.game_engine import cmd_play


def main():
    parser = argparse.ArgumentParser(
        description="Squid Game Drone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("collect", help="Collect motion baseline data")
    p.add_argument("--session", required=True)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--webcam", action="store_true")

    p = sub.add_parser("label", help="Label a captured session")
    p.add_argument("--session", required=True)
    p.add_argument("--label", required=True, choices=["still", "moving"])

    p = sub.add_parser("evaluate", help="Evaluate the motion baseline")
    p.add_argument("--threshold", type=float, default=MOTION_AREA_RATIO_THRESHOLD)

    sub.add_parser("sweep", help="Find a good motion threshold")

    p = sub.add_parser("motion-live", help="Live motion detector test")
    p.add_argument("--webcam", action="store_true")

    p = sub.add_parser("registry-test", help="Standalone player tracking test")
    p.add_argument("--webcam", action="store_true")

    p = sub.add_parser("hats-test", help="Standalone AR hat overlay test")
    p.add_argument("--webcam", action="store_true")

    p = sub.add_parser("register", help="Register a player's face")
    p.add_argument("--name", required=False, default=None, help="Player name")
    p.add_argument("--webcam", action="store_true", help="Use the webcam for registration")
    p.add_argument(
        "--cap-color",
        choices=list(CAP_COLOR_HSV_RANGES.keys()),
        default=None,
        help="Optional cap color used by long-range mode",
    )

    sub.add_parser("players", help="List registered players")
    sub.add_parser("leaderboard", help="Show the leaderboard")

    p = sub.add_parser("check", help="Run a pre-flight safety check")
    p.add_argument("--webcam", action="store_true", help="Skip drone checks in webcam mode")

    p = sub.add_parser("fly-test", help="Simple takeoff, hover, and landing test")
    p.add_argument("--seconds", type=float, default=10.0, help="Hover duration in seconds")

    p = sub.add_parser("play", help="Run the full game")
    p.add_argument("--webcam", action="store_true")
    p.add_argument("--no-flight", action="store_true")
    p.add_argument("--skip-lobby", action="store_true")
    p.add_argument("--no-face-id", action="store_true", help="Disable face recognition for higher speed")
    p.add_argument("--no-voice", action="store_true", help="Disable voice announcements")
    p.add_argument("--no-evidence", action="store_true", help="Disable elimination evidence snapshots")
    p.add_argument(
        "--mode",
        nargs="+",
        choices=GAME_MODE_CHOICES,
        default=["classic"],
        help="Game modes: classic, blindfold, sack-race, long-range",
    )

    args = parser.parse_args()

    if args.command == "collect":
        cmd_collect(args.session, args.duration, use_webcam=args.webcam)
    elif args.command == "label":
        cmd_label(args.session, args.label)
    elif args.command == "evaluate":
        cmd_evaluate(args.threshold)
    elif args.command == "sweep":
        cmd_sweep()
    elif args.command == "motion-live":
        cmd_motion_live(use_webcam=args.webcam)
    elif args.command == "registry-test":
        cmd_registry_test(use_webcam=args.webcam)
    elif args.command == "hats-test":
        cmd_hats_test(use_webcam=args.webcam)
    elif args.command == "register":
        cmd_register(args.name, use_webcam=args.webcam, cap_color=args.cap_color)
    elif args.command == "players":
        cmd_list_players()
    elif args.command == "leaderboard":
        cmd_leaderboard()
    elif args.command == "check":
        cmd_check(use_webcam=args.webcam)
    elif args.command == "fly-test":
        cmd_fly_test(hover_seconds=args.seconds)
    elif args.command == "play":
        cmd_play(
            use_webcam=args.webcam,
            allow_flight=not args.no_flight,
            skip_lobby=args.skip_lobby,
            use_face_id=not args.no_face_id,
            use_voice=not args.no_voice,
            use_evidence=not args.no_evidence,
            modes=args.mode,
        )


if __name__ == "__main__":
    main()
