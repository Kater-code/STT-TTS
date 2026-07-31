"""Альтернативный TTS: ElevenLabs, модель eleven_multilingual_v2.

Смысл держать второй провайдер — клонирование голоса: если нужно, чтобы
тагальская дорожка звучала голосом исходного диктора, это делается здесь,
а не в Google. Тагальский у multilingual_v2 поддерживается, но качество
просодии заметно зависит от голоса — проверяйте на своём материале.
"""

from __future__ import annotations

import os

import numpy as np

from ..config import Config

_API = "https://api.elevenlabs.io/v1/text-to-speech"
_MODEL = "eleven_multilingual_v2"


class ElevenLabsTTS:
    def __init__(self, cfg: Config):
        import requests  # зависимость опциональна

        self._requests = requests
        self.cfg = cfg
        self.sample_rate = 24000  # pcm_24000 — единственный удобный сырой формат
        self.api_key = os.environ["ELEVENLABS_API_KEY"]
        self.voice_id = cfg.tts_voice if cfg.tts_voice.startswith("el:") else os.environ.get(
            "ELEVENLABS_VOICE_ID", ""
        )
        self.voice_id = self.voice_id.removeprefix("el:")
        if not self.voice_id:
            raise RuntimeError("укажите ELEVENLABS_VOICE_ID или tts_voice='el:<voice_id>'")

    def synthesize(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.int16)

        resp = self._requests.post(
            f"{_API}/{self.voice_id}",
            headers={"xi-api-key": self.api_key, "accept": "audio/pcm"},
            params={"output_format": f"pcm_{self.sample_rate}"},
            json={
                "text": text,
                "model_id": _MODEL,
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return np.frombuffer(resp.content, dtype=np.int16)
