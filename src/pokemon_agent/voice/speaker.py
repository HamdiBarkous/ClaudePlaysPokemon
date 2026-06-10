"""Speech pipeline that overlaps synthesis and thinking with playback.

Design (see HOW_IT_WORKS): gameplay is paced by the voice. Per turn the tool
does:

    clip = speaker.submit(say)   # synthesis starts immediately, in background
    clip.started.wait()          # gate: previous line must finish playing
    emulator.press_buttons(...)  # buttons run WHILE this line is spoken

Two worker threads form the pipeline:
- synth thread: turns text into PCM (overlaps the previous clip's playback)
- play thread: plays clips strictly in order, one at a time

So speech never drifts more than one action behind the screen, and the only
dead air is when the model thinks longer than the previous line lasts.
"""

import logging
import queue
import threading

logger = logging.getLogger(__name__)

# A clip that never plays still releases the gate after this long.
GATE_TIMEOUT_SECONDS = 30.0


class Clip:
    """One spoken line moving through the pipeline."""

    def __init__(self, text: str):
        self.text = text
        self.audio = None  # np.ndarray once synthesized
        self.started = threading.Event()  # playback began (or was skipped)
        self.finished = threading.Event()

    def wait_started(self, timeout: float = GATE_TIMEOUT_SECONDS) -> None:
        """Block until this clip starts playing (the tool's gate)."""
        if not self.started.wait(timeout):
            logger.warning("[TTS] Gate timed out waiting for playback to start")


class Speaker:
    """Two-stage TTS pipeline: synthesize in background, play in order.

    Playback uses one persistent OutputStream with blocking writes and high
    latency: per-clip sd.play() opened/closed a stream per sentence (audible
    pops on WSLg's Pulse bridge) and fed audio from a Python callback that
    crackled whenever other threads held the GIL too long.
    """

    def __init__(self, engine):
        self._engine = engine
        self._synth_q: queue.Queue = queue.Queue()
        self._play_q: queue.Queue = queue.Queue()
        self._stream = None
        self._win_player = None
        # WSLg's Pulse bridge stutters during playback (verified: identical
        # wavs play clean on Windows) — route audio to Windows when on WSL
        from pokemon_agent.voice.windows_audio import WindowsAudioPlayer, is_wsl

        if is_wsl():
            try:
                self._win_player = WindowsAudioPlayer()
                logger.info("[TTS] Playing through Windows audio (WSL detected)")
            except Exception as e:
                logger.warning(f"[TTS] Windows audio bridge unavailable ({e}), using Pulse")
        self._synth_thread = threading.Thread(target=self._synth_worker, daemon=True)
        self._play_thread = threading.Thread(target=self._play_worker, daemon=True)
        self._synth_thread.start()
        self._play_thread.start()

    def submit(self, text: str) -> Clip:
        """Queue a line for speaking. Returns immediately; synthesis starts now."""
        clip = Clip(text)
        if not text or not text.strip():
            clip.started.set()
            clip.finished.set()
            return clip
        self._synth_q.put(clip)
        return clip

    def stop(self) -> None:
        """Stop playback and shut down the pipeline."""
        self._synth_q.put(None)
        self._play_q.put(None)
        if self._win_player is not None:
            self._win_player.stop()
        if self._stream is not None:
            try:
                self._stream.abort()
                self._stream.close()
            except Exception:
                pass

    def _synth_worker(self) -> None:
        while True:
            clip = self._synth_q.get()
            if clip is None:
                return
            try:
                clip.audio = self._engine.synthesize(clip.text)
            except Exception as e:
                logger.warning(f"[TTS] Synthesis failed, skipping line: {e}")
            self._play_q.put(clip)

    def _play_worker(self) -> None:
        while True:
            clip = self._play_q.get()
            if clip is None:
                return
            clip.started.set()
            try:
                if clip.audio is not None and len(clip.audio):
                    if self._win_player is not None:
                        self._win_player.play(clip.audio, self._engine.sample_rate)
                    else:
                        self._write_to_stream(clip.audio)
            except Exception as e:
                logger.warning(f"[TTS] Playback failed: {e}")
                self._stream = None  # reopen on the next clip
            finally:
                clip.finished.set()

    def _get_stream(self):
        if self._stream is None:
            import os

            import sounddevice as sd

            # WSLg's Pulse bridge crackles with small buffers; this is the
            # documented knob for the ALSA->Pulse plugin (must be set before
            # the first connection)
            os.environ.setdefault("PULSE_LATENCY_MSEC", "60")
            self._stream = sd.OutputStream(
                samplerate=self._engine.sample_rate,
                channels=1,
                dtype="int16",
                latency="high",
            )
            self._stream.start()
        return self._stream

    def _write_to_stream(self, audio) -> None:
        """Blocking write into the persistent stream, with padded edges.

        The short silence before and after each clip keeps the line's first
        syllable clear of the stream spin-up and masks the underflow boundary
        between clips.
        """
        import numpy as np

        pad = np.zeros(int(0.05 * self._engine.sample_rate), dtype=audio.dtype)
        stream = self._get_stream()
        stream.write(np.concatenate([pad, audio, pad]).reshape(-1, 1))


class NoOpSpeaker:
    """Speaker stand-in when TTS is disabled — the game runs exactly as before."""

    def submit(self, text: str) -> Clip:
        clip = Clip(text)
        clip.started.set()
        clip.finished.set()
        return clip

    def stop(self) -> None:
        pass


def create_speaker(settings):
    """Build the speaker configured in settings, falling back to no-op."""
    if settings.tts_engine == "cartesia":
        if not settings.cartesia_api_key or not settings.cartesia_voice:
            logger.warning(
                "[TTS] tts_engine=cartesia but CARTESIA_API_KEY/CARTESIA_VOICE "
                "not set — running without voice"
            )
            return NoOpSpeaker()
        from pokemon_agent.voice.cartesia import CartesiaEngine

        return Speaker(
            CartesiaEngine(
                api_key=settings.cartesia_api_key,
                voice_id=settings.cartesia_voice,
                model_id=settings.cartesia_model,
            )
        )
    return NoOpSpeaker()
