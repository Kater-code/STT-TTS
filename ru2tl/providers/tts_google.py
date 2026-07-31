"""TTS через Google Cloud Text-to-Speech, локаль fil-PH.

Почему Google: у него самое надёжное покрытие тагальского из мейнстримных
облачных TTS — четыре Standard и четыре WaveNet голоса под fil-PH.
Аутентификация — service account через GOOGLE_APPLICATION_CREDENTIALS.
"""

from __future__ import annotations

import numpy as np

from ..config import Config

_LANGUAGE_CODE = "fil-PH"


class GoogleTTS:
    def __init__(self, cfg: Config):
        from google.cloud import texttospeech  # импорт внутри — зависимость опциональна

        self._tts = texttospeech
        self.cfg = cfg
        self.sample_rate = cfg.tts_sample_rate
        self.client = texttospeech.TextToSpeechClient()
        self._voice = texttospeech.VoiceSelectionParams(
            language_code=_LANGUAGE_CODE,
            name=cfg.tts_voice,
        )
        self._audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate,
            speaking_rate=cfg.tts_speaking_rate,
        )

    def synthesize(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.int16)

        resp = self.client.synthesize_speech(
            input=self._tts.SynthesisInput(text=text),
            voice=self._voice,
            audio_config=self._audio_config,
        )
        # LINEAR16 приходит WAV-контейнером: 44 байта заголовка перед данными.
        data = resp.audio_content
        offset = data.find(b"data")
        offset = offset + 8 if offset != -1 else 44
        return np.frombuffer(data[offset:], dtype=np.int16)

    def list_voices(self) -> list[str]:
        """Диагностика: какие fil-PH голоса доступны проекту."""
        resp = self.client.list_voices(language_code=_LANGUAGE_CODE)
        return [v.name for v in resp.voices]
