"""Фабрики провайдеров. Импорты ленивые: не хотите Google TTS — не ставьте его."""

from __future__ import annotations

from ..config import Config
from .base import ASR, MT, TTS


def make_asr(cfg: Config) -> ASR:
    from .asr_openai import OpenAIWhisperASR

    return OpenAIWhisperASR(cfg)


def make_mt(cfg: Config) -> MT:
    from .mt_openai import OpenAITranslator

    return OpenAITranslator(cfg)


def make_tts(cfg: Config) -> TTS:
    if cfg.tts_provider == "google":
        from .tts_google import GoogleTTS

        return GoogleTTS(cfg)
    if cfg.tts_provider == "elevenlabs":
        from .tts_elevenlabs import ElevenLabsTTS

        return ElevenLabsTTS(cfg)
    raise ValueError(f"неизвестный TTS-провайдер: {cfg.tts_provider!r} (google | elevenlabs)")


__all__ = ["ASR", "MT", "TTS", "make_asr", "make_mt", "make_tts"]
