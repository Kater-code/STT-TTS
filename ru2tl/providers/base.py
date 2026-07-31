"""Контракты провайдеров. Ядро пайплайна знает только про эти три протокола,
поэтому любой из них можно подменить, не трогая pipeline.py.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..models import Segment


class ASR(Protocol):
    def transcribe_file(self, path: str) -> list[Segment]:
        """Файл → сегменты с таймкодами и русским текстом."""

    def transcribe_pcm(self, pcm: np.ndarray, sample_rate: int) -> str:
        """Один кусок PCM16 → русский текст без таймкодов (стриминг)."""


class MT(Protocol):
    def translate_batch(self, segments: list[Segment]) -> list[Segment]:
        """Перевести сегменты пачкой, с общим контекстом. Заполняет .translation."""

    def translate_one(self, text: str, context: list[tuple[str, str]]) -> str:
        """Одна реплика с оглядкой на предыдущие пары (ru, tl)."""


class TTS(Protocol):
    sample_rate: int

    def synthesize(self, text: str) -> np.ndarray:
        """Тагальский текст → PCM16 mono на self.sample_rate."""
