"""Cartesia text-to-speech engine."""

import logging
import time

import httpx
import numpy as np

logger = logging.getLogger(__name__)

CARTESIA_TTS_URL = "https://api.cartesia.ai/tts/bytes"
CARTESIA_VERSION = "2024-11-13"
SAMPLE_RATE = 22050


class CartesiaEngine:
    """Synthesizes speech via Cartesia's bytes endpoint (raw 16-bit PCM)."""

    def __init__(self, api_key: str, voice_id: str, model_id: str = "sonic-2"):
        self._voice_id = voice_id
        self._model_id = model_id
        self._client = httpx.Client(
            headers={
                "X-API-Key": api_key,
                "Cartesia-Version": CARTESIA_VERSION,
            },
            timeout=30.0,
        )

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    def synthesize(self, text: str) -> np.ndarray:
        """Convert text to a mono int16 PCM array.

        Raises on API errors — the Speaker catches and skips the clip so a TTS
        outage never stops the game.
        """
        start = time.monotonic()
        response = self._client.post(
            CARTESIA_TTS_URL,
            json={
                "model_id": self._model_id,
                "transcript": text,
                "voice": {"mode": "id", "id": self._voice_id},
                "output_format": {
                    "container": "raw",
                    "encoding": "pcm_s16le",
                    "sample_rate": SAMPLE_RATE,
                },
                "language": "en",
            },
        )
        response.raise_for_status()
        audio = np.frombuffer(response.content, dtype=np.int16)
        logger.info(
            f"[TTS] Synthesized {len(audio) / SAMPLE_RATE:.1f}s of audio "
            f"in {time.monotonic() - start:.2f}s"
        )
        return audio

    def close(self) -> None:
        self._client.close()
