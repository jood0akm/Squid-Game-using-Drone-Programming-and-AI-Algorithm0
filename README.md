# Squid Game Drone - UltraFast V3

## Recommended Windows launch
Double-click `START_GAME_WEBCAM_FAST.bat`.

Equivalent command:

```powershell
py main.py play --webcam --no-flight --no-evidence
```

This mode still uses Face ID in the lobby so final results contain player names. Face ID is not run inside the active game loop, which keeps the live camera smoother.

## Ultra-fast mode without player names

```powershell
py main.py play --webcam --no-flight --no-face-id --no-evidence
```

or double-click `START_GAME_WEBCAM_ULTRAFAST_NO_NAMES.bat`.

## V3 game flow
- The camera keeps only the newest webcam frame.
- YOLO tracking runs on a background worker and never blocks the display loop.
- Tracking uses a 256-pixel inference size for lower CPU latency.
- RED LIGHT starts with a full-screen eye warning for 3 seconds.
- Motion detection starts after the eye screen, using the live camera.
- A single elimination does not end the game.
- A single winner does not end the game.
- The game ends only when every starting player is either WIN or OUT.
- Winners are ranked by finish order: 1st, 2nd, 3rd, and so on.
- Each WIN/OUT result records the Green/Red round in which it happened.
- The final dashboard shows rank, player name, result, and round.
- A new game starts automatically after the result screen.

## Register a player

```powershell
py main.py register --name "Player Name" --webcam
```

Capture 15 clear face samples using Space.

## List registered players

```powershell
py main.py players
```

## Controls
- `Q`: safe exit
- Lobby: `S` can manually start after players are visible
- Hat selection: mouse click or number keys
