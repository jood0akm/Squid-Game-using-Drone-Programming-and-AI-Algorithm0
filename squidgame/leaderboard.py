"""Persistent win-count leaderboard."""

import json
import os

from .config import LEADERBOARD_FILE


def load_leaderboard() -> dict:
    if not os.path.exists(LEADERBOARD_FILE):
        return {}
    try:
        with open(LEADERBOARD_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_leaderboard(board: dict):
    tmp_path = LEADERBOARD_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(board, f, ensure_ascii=False)
    os.replace(tmp_path, LEADERBOARD_FILE)


def update_leaderboard(winner_names: list):
    if not winner_names:
        return
    board = load_leaderboard()
    for name in winner_names:
        if name.startswith("Player-"):
            continue
        board[name] = board.get(name, 0) + 1
    save_leaderboard(board)


def cmd_leaderboard():
    board = load_leaderboard()
    if not board:
        print("[INFO] No wins have been recorded yet.")
        return
    ranked = sorted(board.items(), key=lambda kv: kv[1], reverse=True)
    print("===== LEADERBOARD =====")
    for i, (name, wins) in enumerate(ranked, start=1):
        prefix = ["1.", "2.", "3."][i - 1] if i <= 3 else f"{i}."
        print(f"{prefix} {name}: {wins} win(s)")
    print("=" * 30)
