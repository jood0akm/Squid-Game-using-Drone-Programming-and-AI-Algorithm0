"""USB-Serial controller for the external ESP32 Red/Green lights.

The game never waits for an acknowledgement from the ESP32. State changes are
queued to a tiny background worker so serial I/O cannot stall the video/game loop.

Optional environment variable:
    SQUID_ESP_PORT=COM5

If not set, the module tries to auto-detect a common ESP32 USB-UART adapter.
"""

import os
import queue
import threading
import time


class TrafficLightController:
    VALID_STATES = {"GREEN", "RED", "OFF"}

    def __init__(self, port=None, baudrate=115200, enabled=True):
        self.enabled = bool(enabled)
        self.baudrate = int(baudrate)
        self.port = port or os.getenv("SQUID_ESP_PORT", "AUTO")
        self._serial = None
        self._queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = None
        self._desired_state = None

        if not self.enabled:
            print("[ESP32] External traffic lights disabled.")
            return

        try:
            import serial
            from serial.tools import list_ports
        except ImportError:
            print("[ESP32] pyserial is not installed. External lights disabled.")
            print("        Install it with: py -m pip install pyserial")
            return

        if self.port == "AUTO":
            self.port = self._auto_find_port(list_ports)

        if not self.port:
            print("[ESP32] No ESP32 serial port found. Game will continue without external lights.")
            print('        Set one manually, for example: $env:SQUID_ESP_PORT="COM5"')
            return

        try:
            self._serial = serial.Serial(
                self.port,
                self.baudrate,
                timeout=0,
                write_timeout=0.15,
            )

            # Many ESP32 boards reset when the serial port opens.
            # This happens once BEFORE gameplay; it does not add delay to
            # Green/Red transitions later.
            time.sleep(1.8)
            try:
                self._serial.reset_input_buffer()
                self._serial.reset_output_buffer()
            except Exception:
                pass

            self._thread = threading.Thread(
                target=self._worker,
                daemon=True,
                name="esp32-traffic-light",
            )
            self._thread.start()

            print(f"[ESP32] Traffic lights connected on {self.port} @ {self.baudrate} baud.")
        except Exception as exc:
            self._serial = None
            print(f"[ESP32] Could not open {self.port}: {exc}")
            print("[ESP32] Game will continue without external lights.")

    @staticmethod
    def _auto_find_port(list_ports_module):
        ports = list(list_ports_module.comports())
        if not ports:
            return None

        keywords = (
            "CP210",
            "CH340",
            "CH341",
            "USB-SERIAL",
            "USB SERIAL",
            "SILICON LABS",
            "ESPRESSIF",
            "UART",
        )

        likely = []
        for p in ports:
            description = f"{p.description} {p.manufacturer or ''}".upper()
            if any(k in description for k in keywords):
                likely.append(p.device)

        if likely:
            print(f"[ESP32] Auto-detected serial port: {likely[0]}")
            return likely[0]

        available = ", ".join(p.device for p in ports)
        print(f"[ESP32] Serial ports found ({available}), but none clearly looks like an ESP32.")
        return None

    @property
    def connected(self):
        return self._serial is not None

    def _worker(self):
        while not self._stop.is_set():
            try:
                state = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if state is None:
                continue

            try:
                self._serial.write((state + "\n").encode("ascii"))
                self._serial.flush()
            except Exception as exc:
                print(f"[ESP32] Serial write warning: {exc}")

    def set_state(self, state):
        state = str(state).upper().strip()
        if state not in self.VALID_STATES:
            raise ValueError(f"Unknown traffic-light state: {state}")

        if self._serial is None:
            return

        if state == self._desired_state:
            return

        self._desired_state = state

        # Keep only the newest requested state. If transitions happen quickly,
        # an obsolete state is discarded instead of being sent late.
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self._queue.put_nowait(state)
        except queue.Full:
            pass

    def green(self):
        self.set_state("GREEN")

    def red(self):
        self.set_state("RED")

    def off(self):
        self.set_state("OFF")

    def close(self):
        if self._serial is None:
            return

        # Send OFF synchronously during shutdown so no lamp is left on.
        try:
            self._serial.write(b"OFF\n")
            self._serial.flush()
            time.sleep(0.03)
        except Exception:
            pass

        self._stop.set()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.5)

        try:
            self._serial.close()
        except Exception:
            pass

        self._serial = None
