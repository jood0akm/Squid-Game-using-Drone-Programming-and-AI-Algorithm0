"""Reliable offline voice announcements using one serialized speech queue."""

import queue
import threading

from .config import VOICE_ENABLED_DEFAULT, VOICE_RATE


class VoiceAnnouncer:
    """Speak queued announcements from one worker thread to avoid pyttsx3 conflicts."""

    def __init__(self, enabled: bool = VOICE_ENABLED_DEFAULT):
        self.enabled = enabled
        self._queue = queue.Queue()
        self._thread = None
        self._ready = threading.Event()

        if not enabled:
            return

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=3.0)

    def _worker(self):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", VOICE_RATE)
            self._ready.set()
        except Exception as exc:
            print(f"[WARN] Voice initialization failed: {exc}")
            print("       Install or repair it with: py -m pip install pyttsx3")
            self.enabled = False
            self._ready.set()
            return

        while True:
            text = self._queue.get()
            if text is None:
                self._queue.task_done()
                break
            try:
                engine.say(str(text))
                engine.runAndWait()
            except Exception as exc:
                print(f"[WARN] Voice announcement failed: {exc}")
            finally:
                self._queue.task_done()

    def say(self, text: str):
        if self.enabled and text:
            self._queue.put(str(text))

    def say_state(self, text: str):
        """Prioritize a Green/Red state announcement over queued non-critical speech."""
        self.clear()
        self.say(text)

    def clear(self):
        if not self.enabled:
            return
        try:
            while True:
                self._queue.get_nowait()
                self._queue.task_done()
        except queue.Empty:
            pass

    def close(self):
        if self._thread is not None and self._thread.is_alive():
            self._queue.put(None)
