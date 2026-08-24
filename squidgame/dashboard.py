"""English documentation."""

import cv2
import numpy as np
import threading

try:
    import winsound

    def play_beep(freq: int = 900, duration_ms: int = 300):
        # Never block the live video loop for a sound effect.
        threading.Thread(target=winsound.Beep, args=(freq, duration_ms), daemon=True).start()
except ImportError:
    def play_beep(freq: int = 900, duration_ms: int = 300):
        print("\a", end="", flush=True)

from .config import (
    TITLE_TEXT,
    TITLE_POP_IN_SEC,
    TITLE_Y_OFFSET,
    HUD_MARGIN,
    PLAYER_LIST_FONT_SCALE,
    BATTERY_LOW_THRESHOLD,
    BATTERY_MED_THRESHOLD,
    GAME_OVER_POLL_MS,
)


COLOR_ALIVE = (0, 210, 0)
COLOR_OUT = (0, 0, 255)
COLOR_WINNER = (0, 220, 255)     
COLOR_GOLD = COLOR_WINNER
COLOR_SILVER = (192, 192, 192)
COLOR_BRONZE = (50, 127, 205)
COLOR_NEW = (210, 210, 210)


_LIST_FONT = cv2.FONT_HERSHEY_COMPLEX | cv2.FONT_ITALIC


def _put_text_with_shadow(frame, text, org, font, scale, color, thickness=2):
    """English documentation."""
    x, y = org
    cv2.putText(frame, text, (x + 1, y + 1), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_battery(frame, battery_percent):
    """English documentation."""
    if battery_percent is None:
        return frame

    if battery_percent <= BATTERY_LOW_THRESHOLD:
        color = (0, 0, 255)
    elif battery_percent <= BATTERY_MED_THRESHOLD:
        color = (0, 220, 255)
    else:
        color = (0, 210, 0)

    label = f"BATT {battery_percent}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.7, 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    x = frame.shape[1] - tw - HUD_MARGIN
    y = HUD_MARGIN + th

    _put_text_with_shadow(frame, label, (x, y), font, scale, color, thickness)
    return frame


def draw_player_list(frame, entries):
    """English documentation."""
    if not entries:
        return frame

    font = _LIST_FONT
    scale = PLAYER_LIST_FONT_SCALE
    thickness = 2
    line_gap = 10

    sizes = [cv2.getTextSize(name, font, scale, thickness)[0] for name, _ in entries]
    line_h = max(h for _, h in sizes) + line_gap

    
    y = frame.shape[0] - HUD_MARGIN
    for (name, color), (tw, th) in reversed(list(zip(entries, sizes))):
        x = frame.shape[1] - tw - HUD_MARGIN
        _put_text_with_shadow(frame, name, (x, y), font, scale, color, thickness)
        y -= line_h

    return frame


def _render_pixel_text_layers(text, base_font_scale=0.62, thickness=2, pixel_size=8):
    """English documentation."""
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, base_font_scale, thickness + 2)
    pad = 8
    w, h = tw + pad * 2, th + baseline + pad * 2
    org = (pad, th + pad - 2)

    color_small = np.zeros((h, w, 3), dtype=np.uint8)
    mask_small = np.zeros((h, w), dtype=np.uint8)

    outline_color = (18, 18, 18)   
    fill_color = (35, 35, 220)     
    highlight_color = (90, 90, 250)  

    
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (2, 2), (-2, 2), (2, -2)]:
        cv2.putText(color_small, text, (org[0] + dx, org[1] + dy), font, base_font_scale,
                    outline_color, thickness + 2, cv2.LINE_8)
        cv2.putText(mask_small, text, (org[0] + dx, org[1] + dy), font, base_font_scale,
                    255, thickness + 2, cv2.LINE_8)

    
    cv2.putText(color_small, text, (org[0], org[1] - 2), font, base_font_scale,
                highlight_color, thickness, cv2.LINE_8)
    cv2.putText(mask_small, text, (org[0], org[1] - 2), font, base_font_scale,
                255, thickness, cv2.LINE_8)

    
    cv2.putText(color_small, text, org, font, base_font_scale, fill_color, thickness, cv2.LINE_8)
    cv2.putText(mask_small, text, org, font, base_font_scale, 255, thickness, cv2.LINE_8)

    big_color = cv2.resize(color_small, (w * pixel_size, h * pixel_size), interpolation=cv2.INTER_NEAREST)
    big_mask = cv2.resize(mask_small, (w * pixel_size, h * pixel_size), interpolation=cv2.INTER_NEAREST)
    return big_color, big_mask


def draw_pixel_logo(frame, elapsed_sec, text=None):
    """English documentation."""
    text = text or TITLE_TEXT
    if elapsed_sec < 0:
        return frame

    fw_frame = frame.shape[1]
    max_logo_w = int(fw_frame * 0.75)  

    base_pixel_size = 9
    _, probe_mask = _render_pixel_text_layers(text, pixel_size=base_pixel_size)
    if probe_mask.shape[1] > max_logo_w:
        ratio = max_logo_w / probe_mask.shape[1]
        base_pixel_size = max(2, int(base_pixel_size * ratio))

    if elapsed_sec < TITLE_POP_IN_SEC:
        
        
        t = elapsed_sec / TITLE_POP_IN_SEC
        ease = 1 - (1 - t) ** 3  
        overshoot = 0.6 * (1 - ease)
        scale = 1.0 + (2.6 * (1 - ease)) + overshoot
        alpha = min(1.0, 0.5 + t)
    else:
        
        scale = 1.0
        alpha = 1.0

    scale = max(0.3, scale)
    pixel_size = max(2, int(round(base_pixel_size * scale)))
    big_color, big_mask = _render_pixel_text_layers(text, pixel_size=pixel_size)

    fh, fw = frame.shape[:2]
    lh, lw = big_mask.shape[:2]
    x0 = (fw - lw) // 2
    y0 = HUD_MARGIN

    
    src_x0, src_y0 = max(0, -x0), max(0, -y0)
    dst_x0, dst_y0 = max(0, x0), max(0, y0)
    src_x1 = src_x0 + min(lw - src_x0, fw - dst_x0)
    src_y1 = src_y0 + min(lh - src_y0, fh - dst_y0)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return frame

    region_mask = big_mask[src_y0:src_y1, src_x0:src_x1] > 0
    region_color = big_color[src_y0:src_y1, src_x0:src_x1]
    dst_x1, dst_y1 = dst_x0 + (src_x1 - src_x0), dst_y0 + (src_y1 - src_y0)

    overlay = frame.copy()
    roi = overlay[dst_y0:dst_y1, dst_x0:dst_x1]
    roi[region_mask] = region_color[region_mask]
    overlay[dst_y0:dst_y1, dst_x0:dst_x1] = roi

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)
    return frame


def draw_eyes(frame, openness: float):
    """Draw the doll-eye state as an overlay while keeping the live camera visible."""
    openness = max(0.0, min(1.0, openness))
    h, w = frame.shape[:2]

    panel_w = min(210, max(150, w // 3))
    panel_h = min(92, max(72, h // 4))
    x1 = max(8, w - panel_w - 12)
    y1 = 68
    x2 = min(w - 8, x1 + panel_w)
    y2 = min(h - 8, y1 + panel_h)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, dst=frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (230, 230, 230), 1)

    label = "EYES OPEN" if openness >= 0.92 else ("EYES CLOSED" if openness <= 0.08 else "EYES OPENING")
    cv2.putText(frame, label, (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (255, 255, 255), 1, cv2.LINE_AA)

    eye_y = y1 + int(panel_h * 0.62)
    eye_w = max(28, panel_w // 5)
    max_eye_h = max(18, int(eye_w * 0.58))
    centers = (x1 + panel_w // 3, x1 + (2 * panel_w) // 3)
    cur_h = max(2, int(max_eye_h * openness))

    for cx in centers:
        if openness < 0.08:
            cv2.line(frame, (cx - eye_w // 2, eye_y), (cx + eye_w // 2, eye_y),
                     (240, 240, 240), 3, cv2.LINE_AA)
        else:
            cv2.ellipse(frame, (cx, eye_y), (eye_w // 2, max(2, cur_h // 2)), 0, 0, 360,
                        (240, 240, 240), -1, cv2.LINE_AA)
            pupil_r = max(2, min(7, cur_h // 4))
            cv2.circle(frame, (cx, eye_y), pupil_r, (10, 10, 10), -1, cv2.LINE_AA)

    return frame


def build_live_entries(registry, eliminated_ids, arrival_order, now):
    """English documentation."""
    rank_colors = {0: COLOR_GOLD, 1: COLOR_SILVER, 2: COLOR_BRONZE}
    arrived_rank = {tid: i for i, tid in enumerate(arrival_order)}

    entries = []
    for tid, rec in registry.players.items():
        if not rec.is_confirmed(now):
            continue
        name = rec.name if rec.name != "Unknown" else f"Player {tid}"
        if tid in eliminated_ids:
            color = COLOR_OUT
        elif tid in arrived_rank:
            rank = arrived_rank[tid]
            color = rank_colors.get(rank, COLOR_ALIVE)
            name = f"{rank + 1}. {name}"
        else:
            color = COLOR_ALIVE
        entries.append((name, color))
    return entries


def _draw_centered_banner(frame, lines, colors, sub_lines=None):
    """English documentation."""
    overlay = frame.copy()
    font = cv2.FONT_HERSHEY_TRIPLEX
    scale, thickness = 1.6, 3

    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    total_h = sum(h for _, h in sizes) + (len(lines) - 1) * 20
    y = (frame.shape[0] - total_h) // 2

    
    pad = 40
    max_w = max(w for w, _ in sizes) if sizes else 0
    cv2.rectangle(overlay, (frame.shape[1] // 2 - max_w // 2 - pad, y - pad),
                  (frame.shape[1] // 2 + max_w // 2 + pad, y + total_h + pad + (60 if sub_lines else 0)),
                  (15, 15, 15), -1)

    for line, color, (tw, th) in zip(lines, colors, sizes):
        x = (frame.shape[1] - tw) // 2
        y += th
        cv2.putText(overlay, line, (x + 2, y + 2), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(overlay, line, (x, y), font, scale, color, thickness, cv2.LINE_AA)
        y += 20

    if sub_lines:
        sub_font, sub_scale, sub_thick = cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2
        y += 20
        for sub in sub_lines:
            (sw, sh), _ = cv2.getTextSize(sub, sub_font, sub_scale, sub_thick)
            x = (frame.shape[1] - sw) // 2
            y += sh
            cv2.putText(overlay, sub, (x, y), sub_font, sub_scale, (255, 255, 255), sub_thick, cv2.LINE_AA)
            y += 14

    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, dst=frame)
    return frame


def draw_game_over(frame):
    """English documentation."""
    return _draw_centered_banner(
        frame,
        lines=["GAME OVER"],
        colors=[(0, 0, 255)],
        sub_lines=["Next round starts automatically", "Press Q to Quit"],
    )


def draw_winner_banner(frame, winner_name):
    """English documentation."""
    return _draw_centered_banner(
        frame,
        lines=["WINNER", winner_name],
        colors=[COLOR_GOLD, (255, 255, 255)],
        sub_lines=["Next round starts automatically", "Press Q to Quit"],
    )


def draw_continue_prompt(frame):
    """English documentation."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.65, 2
    text = "Next round starts automatically   |   Press Q to Quit"
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (frame.shape[1] - tw) // 2
    y = frame.shape[0] - HUD_MARGIN - 10
    _put_text_with_shadow(frame, text, (x, y), font, scale, (255, 255, 255), thickness)
    return frame


def build_ranked_entries(registry, eliminated_ids, winner_order, now):
    """English documentation."""
    rank_colors = {0: COLOR_GOLD, 1: COLOR_SILVER, 2: COLOR_BRONZE}
    winner_set = set(winner_order)

    entries = []
    for rank, tid in enumerate(winner_order):
        rec = registry.players.get(tid)
        name = rec.name if rec and rec.name != "Unknown" else f"Player {tid}"
        color = rank_colors.get(rank, COLOR_ALIVE)
        entries.append((f"{rank + 1}. {name}", color))

    for tid, rec in registry.players.items():
        if tid in winner_set or not rec.is_confirmed(now):
            continue
        name = rec.name if rec.name != "Unknown" else f"Player {tid}"
        entries.append((name, COLOR_OUT))

    return entries


def draw_fullscreen_eyes(frame, openness: float, remaining_sec: float, game_number: int, round_number: int):
    """Replace the camera view with a full-frame doll-eye warning before RED LIGHT detection starts."""
    openness = max(0.0, min(1.0, openness))
    h, w = frame.shape[:2]
    screen = np.zeros_like(frame)
    screen[:] = (10, 10, 10)

    cv2.putText(screen, f"GAME {game_number}  |  ROUND {round_number}", (22, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (225, 225, 225), 2, cv2.LINE_AA)

    title = "RED LIGHT"
    title_scale = max(1.1, min(2.2, w / 350.0))
    title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_TRIPLEX, title_scale, 4)[0]
    cv2.putText(screen, title, ((w - title_size[0]) // 2, max(72, int(h * 0.19))),
                cv2.FONT_HERSHEY_TRIPLEX, title_scale, (0, 0, 245), 4, cv2.LINE_AA)

    eye_y = int(h * 0.55)
    eye_w = max(85, int(w * 0.20))
    eye_h = max(42, int(h * 0.20 * openness))
    centers = (int(w * 0.31), int(w * 0.69))

    for cx in centers:
        if openness <= 0.05:
            cv2.line(screen, (cx - eye_w // 2, eye_y), (cx + eye_w // 2, eye_y),
                     (245, 245, 245), 8, cv2.LINE_AA)
        else:
            cv2.ellipse(screen, (cx, eye_y), (eye_w // 2, max(4, eye_h // 2)),
                        0, 0, 360, (245, 245, 245), -1, cv2.LINE_AA)
            pupil_r = max(8, min(24, eye_h // 4))
            cv2.circle(screen, (cx, eye_y), pupil_r, (12, 12, 12), -1, cv2.LINE_AA)
            cv2.circle(screen, (cx - pupil_r // 3, eye_y - pupil_r // 3),
                       max(2, pupil_r // 5), (255, 255, 255), -1, cv2.LINE_AA)

    status = "EYES OPENING - FREEZE NOW" if openness < 0.98 else "EYES OPEN - DO NOT MOVE"
    scale = 0.76 if w >= 600 else 0.62
    size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
    cv2.putText(screen, status, ((w - size[0]) // 2, int(h * 0.82)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)

    countdown = f"LIVE CAMERA IN {max(0.0, remaining_sec):.1f}s"
    size = cv2.getTextSize(countdown, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)[0]
    cv2.putText(screen, countdown, ((w - size[0]) // 2, int(h * 0.92)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (180, 180, 180), 2, cv2.LINE_AA)
    return screen


def draw_results_dashboard(frame, result_rows: list, game_number: int):
    """Draw final game results with Restart and Quit buttons."""

    h, w = frame.shape[:2]

    screen = np.zeros_like(frame)
    screen[:] = (18, 18, 22)

    # =========================
    # TITLE
    # =========================

    cv2.rectangle(
        screen,
        (0, 0),
        (w, 64),
        (28, 28, 34),
        -1
    )

    title = f"GAME {game_number} RESULTS"

    cv2.putText(
        screen,
        title,
        (22, 42),
        cv2.FONT_HERSHEY_TRIPLEX,
        0.90,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # =========================
    # TABLE HEADERS
    # =========================

    cols = [
        int(w * 0.03),
        int(w * 0.14),
        int(w * 0.43),
        int(w * 0.73),
        int(w * 0.86)
    ]

    headers = [
        "RANK",
        "PLAYER",
        "EVIDENCE",
        "RESULT",
        "ROUND"
    ]

    header_y = 94

    for x, header in zip(cols, headers):

        cv2.putText(
            screen,
            header,
            (x, header_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (165, 165, 175),
            1,
            cv2.LINE_AA
        )

    cv2.line(
        screen,
        (18, 106),
        (w - 18, 106),
        (75, 75, 82),
        1
    )

    # =========================
    # RESULTS
    # =========================

    # Reserve space at the bottom for buttons
    available_h = max(
        80,
        h - 200
    )

    row_h = max(
        22,
        min(
            34,
            available_h // max(1, len(result_rows))
        )
    )

    font_scale = (
        0.58
        if row_h >= 28
        else 0.48
    )

    y = 106 + row_h

    rank_labels = {
        1: "1ST",
        2: "2ND",
        3: "3RD"
    }

    for row in result_rows:

        status = row.get(
            "status",
            "OUT"
        )

        rank = row.get("rank")

        rank_text = rank_labels.get(
            rank,
            str(rank) if rank else "-"
        )

        name = str(
            row.get(
                "name",
                "Player"
            )
        )

        result_text = (
            "WIN"
            if status == "WIN"
            else "OUT"
        )

        round_text = str(
            row.get(
                "round",
                "-"
            )
        )

        evidence_path = row.get(
            "evidence"
        )

        if status == "WIN":

            if rank == 1:
                color = COLOR_GOLD

            elif rank == 2:
                color = COLOR_SILVER

            elif rank == 3:
                color = COLOR_BRONZE

            else:
                color = COLOR_ALIVE

        else:
            color = COLOR_OUT

        if y > h - 90:
            break

        cv2.putText(
            screen,
            rank_text,
            (cols[0], y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            screen,
            name[:16],
            (cols[1], y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (245, 245, 245),
            2,
            cv2.LINE_AA
        )

        if evidence_path:
            evidence_image = cv2.imread(evidence_path)

            if evidence_image is not None:
                thumb_h = max(20, row_h - 6)
                thumb_w = max(30, int(thumb_h * 1.45))
                evidence_image = cv2.resize(
                    evidence_image,
                    (thumb_w, thumb_h),
                    interpolation=cv2.INTER_AREA
                )

                thumb_x = cols[2]
                thumb_y = max(108, y - row_h + 3)
                end_x = min(w, thumb_x + thumb_w)
                end_y = min(h, thumb_y + thumb_h)

                visible_w = end_x - thumb_x
                visible_h = end_y - thumb_y

                if visible_w > 0 and visible_h > 0:
                    screen[
                        thumb_y:end_y,
                        thumb_x:end_x
                    ] = evidence_image[
                        :visible_h,
                        :visible_w
                    ]

                    cv2.rectangle(
                        screen,
                        (thumb_x, thumb_y),
                        (end_x, end_y),
                        color,
                        1
                    )

        cv2.putText(
            screen,
            result_text,
            (cols[3], y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            screen,
            round_text,
            (cols[4], y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (230, 230, 230),
            2,
            cv2.LINE_AA
        )

        y += row_h

    # =========================
    # BUTTONS
    # =========================

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

    # Restart button
    restart_x1 = start_x
    restart_x2 = (
        restart_x1
        + button_w
    )

    cv2.rectangle(
        screen,
        (restart_x1, button_y),
        (restart_x2, button_y + button_h),
        (40, 150, 70),
        -1
    )

    cv2.rectangle(
        screen,
        (restart_x1, button_y),
        (restart_x2, button_y + button_h),
        (255, 255, 255),
        1
    )

    restart_text = "RESTART"

    text_size = cv2.getTextSize(
        restart_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        2
    )[0]

    cv2.putText(
        screen,
        restart_text,
        (
            restart_x1
            + (
                button_w
                - text_size[0]
            ) // 2,
            button_y + 29
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Quit button
    quit_x1 = (
        restart_x2
        + gap
    )

    quit_x2 = (
        quit_x1
        + button_w
    )

    cv2.rectangle(
        screen,
        (quit_x1, button_y),
        (quit_x2, button_y + button_h),
        (45, 45, 180),
        -1
    )

    cv2.rectangle(
        screen,
        (quit_x1, button_y),
        (quit_x2, button_y + button_h),
        (255, 255, 255),
        1
    )

    quit_text = "QUIT"

    text_size = cv2.getTextSize(
        quit_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        2
    )[0]

    cv2.putText(
        screen,
        quit_text,
        (
            quit_x1
            + (
                button_w
                - text_size[0]
            ) // 2,
            button_y + 29
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return screen
