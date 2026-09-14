"""Speech-to-text engine for ARGUS — GigaAM (gigastt) + Vosk fallback.

v0.10.0 — Two wake-words edition
--------------------------------
* Wake-word "аргус" → mode = "system"  → fast-path
* Wake-word "вопрос" → mode = "question" → LLM
* command_recognized(text, mode)
"""

from __future__ import annotations

import io
import json
import queue
import re
import threading
import time
import wave
from pathlib import Path
from typing import Any, List, Optional, Tuple

from PySide6.QtCore import QObject, Signal

from core.logger import get_logger

logger = get_logger(__name__)

try:
    import sounddevice as sd
    _SD_IMPORTED = True
except ImportError:
    sd = None  # type: ignore
    _SD_IMPORTED = False

try:
    import vosk
    _VOSK_IMPORTED = True
except ImportError:
    vosk = None  # type: ignore
    _VOSK_IMPORTED = False

try:
    from core.voice.gigastt_client import GigaSTTClient
    _GIGASTT_IMPORTED = True
except ImportError:
    GigaSTTClient = None  # type: ignore
    _GIGASTT_IMPORTED = False


SAMPLE_RATE = 16000
BLOCK_SIZE = 4000

SYSTEM_WAKE_WORDS = ["аргус", "argus"]
QUESTION_WAKE_WORDS = ["вопрос", "вопроса", "вопросы"]
DEFAULT_WAKE_WORDS = SYSTEM_WAKE_WORDS + QUESTION_WAKE_WORDS

_PUNCT_RE = re.compile(r"[^\w\s\-]+", re.UNICODE)


class VoiceInput(QObject):
    wake_detected = Signal()
    listening_started = Signal()
    listening_stopped = Signal()
    command_recognized = Signal(str, str)
    error_occurred = Signal(str)
    state_changed = Signal(str)

    def __init__(
        self,
        enabled: bool = True,
        engine: str = "gigastt",
        model_dir: Optional[Path] = None,
        wake_words: Optional[List[str]] = None,
        listen_timeout_seconds: float = 8.0,
        silence_seconds: float = 1.0,
        dialog_window_seconds: float = 60.0,
        threshold: int = 500,
        device_index: Optional[int] = None,
        voice_output: Optional[Any] = None,
        gigastt_url: str = "http://127.0.0.1:9876",
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.enabled = bool(enabled)
        self.engine = (engine or "gigastt").strip().lower()
        self.model_dir = Path(model_dir) if model_dir else None
        self.system_wake_words = [w.lower() for w in SYSTEM_WAKE_WORDS]
        self.question_wake_words = [w.lower() for w in QUESTION_WAKE_WORDS]
        if wake_words:
            for w in wake_words:
                wl = w.lower()
                if wl not in self.system_wake_words:
                    self.system_wake_words.append(wl)
        self.wake_words = self.system_wake_words + self.question_wake_words
        self.listen_timeout_seconds = float(listen_timeout_seconds)
        self.silence_seconds = float(silence_seconds)
        self.dialog_window_seconds = float(dialog_window_seconds)
        self.threshold = int(threshold)
        self.device_index = device_index
        self.voice_output = voice_output
        self.gigastt_url = gigastt_url
        self.gigastt: Optional[GigaSTTClient] = None
        self._gigastt_ok = False

        self._state = "idle"
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._audio_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=64)

        self._vosk_model: Optional[Any] = None
        self._vosk_recognizer: Optional[Any] = None
        self._stream: Optional[Any] = None
        self._dialog_until: float = 0.0

        self._pcm_buffer: List[bytes] = []
        self._buffering = False
        self._capture_started_at: float = 0.0

        self._paused = False
        self._active_mode: str = "system"
        self._next_mode_override: Optional[str] = None

        if not _SD_IMPORTED:
            logger.warning("sounddevice not installed")
            self._backend_ok = False
        else:
            self._backend_ok = self._check_backends()

    @classmethod
    def from_config(cls, config: Any, voice_output: Optional[Any] = None) -> "VoiceInput":
        model_dir_raw = config.get_str("voice.input.model_dir", "models/vosk/ru")
        try:
            base = config.get_str("paths.base_dir", "")
            model_dir = Path(base) / model_dir_raw if base else Path(model_dir_raw)
        except Exception:
            model_dir = Path(model_dir_raw)

        wake_words = config.get("voice.input.wake_words", None)
        if not isinstance(wake_words, list):
            wake_words = None

        device_index = config.get_int("voice.input.device_index", -1)

        return cls(
            enabled=config.get_bool("voice.input.enabled", True),
            engine=config.get_str("voice.input.engine", "gigastt"),
            model_dir=model_dir,
            wake_words=wake_words,
            listen_timeout_seconds=config.get_float("voice.input.listen_timeout_seconds", 8.0),
            silence_seconds=config.get_float("voice.input.silence_seconds", 1.0),
            dialog_window_seconds=config.get_float("voice.input.dialog_window_seconds", 60.0),
            threshold=config.get_int("voice.input.threshold", 500),
            device_index=device_index if device_index >= 0 else None,
            voice_output=voice_output,
            gigastt_url=config.get_str("voice.input.gigastt_url", "http://127.0.0.1:9876"),
        )

    def _check_backends(self) -> bool:
        if not _VOSK_IMPORTED:
            logger.warning("vosk not installed")
        if self.model_dir is None or not self.model_dir.exists():
            logger.warning("Vosk model dir not found: %s", self.model_dir)
            return False
        needed = ["am", "conf", "graph"]
        return all((self.model_dir / p).exists() for p in needed)

    @property
    def is_listening(self) -> bool:
        return self._state in ("listening", "capturing", "dialog")

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def active_mode(self) -> str:
        return self._active_mode

    def set_enabled(self, flag: bool) -> bool:
        self.enabled = bool(flag)
        self._paused = not self.enabled
        logger.info("Voice input enabled=%s (paused=%s)", self.enabled, self._paused)
        return self.enabled

    def start_question_mode(self) -> None:
        self._next_mode_override = "question"
        self._dialog_until = time.time() + self.dialog_window_seconds
        logger.info("Question mode armed")

    def start(self) -> bool:
        if not self._backend_ok:
            return False
        if self._thread and self._thread.is_alive():
            return True

        try:
            logger.info("Loading Vosk model from %s ...", self.model_dir)
            import contextlib, io as _io
            with contextlib.redirect_stderr(_io.StringIO()):
                self._vosk_model = vosk.Model(str(self.model_dir))
            self._vosk_recognizer = vosk.KaldiRecognizer(self._vosk_model, SAMPLE_RATE)
            self._vosk_recognizer.SetWords(False)
            logger.info("Vosk model loaded (for wake-word)")
        except Exception as exc:
            logger.error("Vosk model load failed: %s", exc)
            self.error_occurred.emit(f"Vosk model load failed: {exc}")
            return False

        if self.engine == "gigastt" and _GIGASTT_IMPORTED:
            try:
                self.gigastt = GigaSTTClient(base_url=self.gigastt_url)
                self._gigastt_ok = self.gigastt.is_alive()
                if self._gigastt_ok:
                    info = self.gigastt.info()
                    logger.info(
                        "gigastt OK — GigaAM %s (v%s, punct=%s, itn=%s)",
                        info.get("model", "?"), info.get("version", "?"),
                        info.get("punctuation"), info.get("itn"),
                    )
                else:
                    logger.warning("gigastt not reachable — Vosk fallback only")
            except Exception as exc:
                logger.error("gigastt init failed: %s", exc)
                self.gigastt = None
                self._gigastt_ok = False

        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._run_loop, name="voice-input", daemon=True)
        self._thread.start()
        self._set_state("listening")
        logger.info(
            "Voice input started (engine=%s gigastt_ok=%s wake_words=%s dialog=%.0fs)",
            self.engine, self._gigastt_ok, self.wake_words, self.dialog_window_seconds,
        )
        return True

    def stop(self) -> None:
        self._stop_flag.set()
        try:
            self._audio_queue.put_nowait(b"")
        except queue.Full:
            pass
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=3.0)
        self._thread = None
        self._close_stream()
        self._set_state("idle")
        if self.gigastt is not None:
            try:
                self.gigastt.close()
            except Exception:
                pass
            self.gigastt = None
            self._gigastt_ok = False
        logger.info("Voice input stopped")

    def shutdown(self) -> None:
        self.stop()
        self._vosk_model = None
        self._vosk_recognizer = None

    def notify_tts_finished(self) -> None:
        self._dialog_until = time.time() + self.dialog_window_seconds

    def _set_state(self, new_state: str) -> None:
        if new_state == self._state:
            return
        self._state = new_state
        try:
            self.state_changed.emit(new_state)
        except Exception as exc:
            logger.debug("state_changed emit failed: %s", exc)

    def _tts_active(self) -> bool:
        try:
            if self.voice_output is None:
                return False
            return bool(getattr(self.voice_output, "is_speaking", False))
        except Exception:
            return False

    def _in_dialog_window(self) -> bool:
        return time.time() < self._dialog_until

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            logger.debug("audio callback status: %s", status)
        if self._stop_flag.is_set():
            return
        if self._tts_active():
            return
        raw = bytes(indata)
        try:
            self._audio_queue.put_nowait(raw)
        except queue.Full:
            try:
                self._audio_queue.get_nowait()
                self._audio_queue.put_nowait(raw)
            except queue.Empty:
                pass

    def _open_stream(self) -> bool:
        try:
            self._stream = sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
                device=self.device_index,
                dtype="int16",
                channels=1,
                callback=self._audio_callback,
            )
            self._stream.start()
            logger.info("Microphone stream opened (device=%s, rate=%s)",
                        self.device_index if self.device_index is not None else "default",
                        SAMPLE_RATE)
            return True
        except Exception as exc:
            logger.error("Cannot open microphone: %s", exc)
            self.error_occurred.emit(f"Microphone error: {exc}")
            return False

    def _close_stream(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is None:
            return
        try:
            stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass

    def _pcm_to_wav(self, chunks: List[bytes]) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            for c in chunks:
                wf.writeframes(c)
        return buf.getvalue()

    def _finalize_capture(self, vosk_text: str = "") -> None:
        final_text = ""

        if self._pcm_buffer and self._gigastt_ok and self.gigastt is not None:
            try:
                wav_bytes = self._pcm_to_wav(self._pcm_buffer)
                t0 = time.time()
                giga_text = self.gigastt.transcribe_wav_bytes(wav_bytes)
                dt = (time.time() - t0) * 1000
                if giga_text:
                    logger.info("GigaAM command (%.0f ms, %d bytes): %s",
                                dt, len(wav_bytes), giga_text)
                    final_text = giga_text
                else:
                    logger.warning("gigastt returned empty (%.0f ms)", dt)
            except Exception as exc:
                logger.error("gigastt transcribe failed: %s", exc)

        if not final_text and vosk_text and not self._is_only_wake_word(vosk_text):
            logger.info("Vosk command (fallback): %s", vosk_text)
            final_text = vosk_text

        self._pcm_buffer.clear()
        self._buffering = False
        self._capture_started_at = 0.0

        mode = self._active_mode
        self._active_mode = "system"

        if final_text:
            logger.info("command_recognized(text='%s', mode='%s')", final_text, mode)
            self.command_recognized.emit(final_text, mode)
        else:
            logger.info("Empty/duplicate command after wake word: '%s'", vosk_text)

        self._set_state("listening")
        self.listening_stopped.emit()
        try:
            self._vosk_recognizer.Reset()
        except Exception:
            pass

    def _run_loop(self) -> None:
        if not self._open_stream():
            self._set_state("idle")
            return

        while not self._stop_flag.is_set():
            if self._tts_active():
                time.sleep(0.05)
                continue

            if self._state == "dialog" and not self._in_dialog_window():
                self._set_state("listening")

            if self._state == "capturing" and self._capture_started_at > 0:
                elapsed = time.time() - self._capture_started_at
                if elapsed > self.listen_timeout_seconds:
                    logger.info("Command timeout after %.1fs", elapsed)
                    self._finalize_capture("")
                    continue

            try:
                chunk = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if not chunk:
                continue

            if self._paused:
                continue

            if self._buffering:
                self._pcm_buffer.append(chunk)

            try:
                is_final = self._vosk_recognizer.AcceptWaveform(chunk)
            except Exception as exc:
                logger.error("Vosk AcceptWaveform failed: %s", exc)
                continue

            if is_final:
                text = self._extract_text(self._vosk_recognizer.Result()).strip()

                if self._state == "dialog" and self._in_dialog_window():
                    if text and not self._is_only_wake_word(text):
                        logger.info("Dialog command (vosk): %s", text)
                        self.command_recognized.emit(text, "system")
                    try:
                        self._vosk_recognizer.Reset()
                    except Exception:
                        pass
                    continue

                if self._state == "capturing":
                    self._finalize_capture(text)
                    continue

                detected = self._detect_wake_word(text)
                if detected:
                    mode, ww = detected
                    logger.info("Wake word detected (final): '%s' → mode=%s", ww, mode)
                    self._start_capture(mode)
                    continue

            else:
                try:
                    partial = self._extract_partial(self._vosk_recognizer.PartialResult())
                except Exception:
                    partial = ""

                if self._state == "listening" and partial:
                    detected = self._detect_wake_word(partial)
                    if detected:
                        mode, ww = detected
                        logger.info("Wake word detected (partial): '%s' → mode=%s", ww, mode)
                        self._start_capture(mode)
                        continue

        self._close_stream()
        self._set_state("idle")

    def _start_capture(self, mode: str) -> None:
        if self._next_mode_override is not None:
            mode = self._next_mode_override
            self._next_mode_override = None
        self._active_mode = mode
        self.wake_detected.emit()
        self._set_state("capturing")
        self.listening_started.emit()
        self._capture_started_at = time.time()
        self._pcm_buffer.clear()
        self._buffering = True
        try:
            self._vosk_recognizer.Reset()
        except Exception:
            pass

    @staticmethod
    def _extract_text(result_json: str) -> str:
        try:
            data = json.loads(result_json)
        except Exception:
            return ""
        return str(data.get("text", "")).strip()

    @staticmethod
    def _extract_partial(result_json: str) -> str:
        try:
            data = json.loads(result_json)
        except Exception:
            return ""
        return str(data.get("partial", "")).strip()

    def _detect_wake_word(self, text: str) -> Optional[Tuple[str, str]]:
        normalized = _PUNCT_RE.sub(" ", text.lower())
        for w in self.question_wake_words:
            if w and w in normalized:
                return ("question", w)
        for w in self.system_wake_words:
            if w and w in normalized:
                return ("system", w)
        return None

    def _is_only_wake_word(self, text: str) -> bool:
        normalized = _PUNCT_RE.sub(" ", text.lower()).strip()
        if not normalized:
            return True
        tokens = normalized.split()
        if not tokens:
            return True
        for tok in tokens:
            matched = False
            for w in self.wake_words:
                if w and (tok == w or tok == w + "s" or tok == w + "ы"):
                    matched = True
                    break
            if not matched:
                return False
        return True
