"""Safe JSON storage helpers for players and cap colors."""

import json
import os

from .config import PLAYERS_FILE, CAP_COLOR_FILE


def load_players() -> dict:
    if not os.path.exists(PLAYERS_FILE):
        return {}
    try:
        with open(PLAYERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_players(players: dict):
    tmp_path = PLAYERS_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(players, f, ensure_ascii=False)
    os.replace(tmp_path, PLAYERS_FILE)
    print(f"[DEBUG] Saved players to: {os.path.abspath(PLAYERS_FILE)}")


def load_cap_colors() -> dict:
    if not os.path.exists(CAP_COLOR_FILE):
        return {}
    try:
        with open(CAP_COLOR_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_cap_colors(mapping: dict):
    tmp_path = CAP_COLOR_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False)
    os.replace(tmp_path, CAP_COLOR_FILE)
