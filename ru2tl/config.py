"""Конфигурация пайплайна: русская речь на входе → тагальская речь на выходе."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Config:
    # --- языки ---
    source_lang: str = "ru"          # ISO-639-1, передаётся в Whisper
    target_lang: str = "tl"
    target_lang_name: str = "Tagalog (Filipino)"

    # --- ASR (OpenAI Whisper) ---
    # whisper-1 — единственная модель, отдающая посегментные таймкоды
    # (response_format="verbose_json"), поэтому она стоит для батча.
    asr_model: str = field(default_factory=lambda: _env("RU2TL_ASR_MODEL", "whisper-1"))
    # Для стриминга таймкоды не нужны — берём модель побыстрее.
    asr_stream_model: str = field(
        default_factory=lambda: _env("RU2TL_ASR_STREAM_MODEL", "gpt-4o-mini-transcribe")
    )
    # Подсказка Whisper: имена, термины, характерная лексика. Улучшает распознавание.
    asr_prompt: str = ""

    # --- MT (OpenAI chat) ---
    mt_model: str = field(default_factory=lambda: _env("RU2TL_MT_MODEL", "gpt-4o"))
    mt_temperature: float = 0.2
    # Сколько сегментов переводить за один запрос в батч-режиме.
    mt_batch_size: int = 40
    # Сколько предыдущих пар (ru → tl) подмешивать как контекст в стриминге.
    mt_stream_context: int = 3

    # --- TTS ---
    tts_provider: str = field(default_factory=lambda: _env("RU2TL_TTS_PROVIDER", "google"))
    # Google fil-PH: Standard-A..D, Wavenet-A..D (A/C — женские, B/D — мужские)
    tts_voice: str = field(default_factory=lambda: _env("RU2TL_TTS_VOICE", "fil-PH-Wavenet-A"))
    tts_sample_rate: int = 24000
    tts_speaking_rate: float = 1.0

    # --- дубляж ---
    # Подгонять длительность каждой реплики под тайминг оригинала
    # (ускорение через ffmpeg atempo, не более fit_max_tempo).
    fit_timing: bool = True
    fit_max_tempo: float = 1.35

    # --- качество перевода ---
    # Глоссарий: {"русский термин": "тагальский эквивалент"}
    glossary: dict[str, str] = field(default_factory=dict)
    # Свободный текст: тон, аудитория, формальность, что не переводить.
    style_guide: str = (
        "Neutral-but-warm conversational register, as used in Philippine broadcast "
        "and online media. Natural Taglish is acceptable where a pure Tagalog word "
        "would sound stilted (e.g. technology and business terms). "
        "Keep proper names, brand names and numbers unchanged."
    )

    @classmethod
    def load(cls, glossary_path: str | os.PathLike[str] | None = None, **overrides) -> "Config":
        cfg = cls(**overrides)
        if glossary_path:
            data = json.loads(Path(glossary_path).read_text(encoding="utf-8"))
            cfg.glossary = data.get("glossary", data)
            cfg.style_guide = data.get("style_guide", cfg.style_guide)
            cfg.asr_prompt = data.get("asr_prompt", cfg.asr_prompt)
        return cfg
