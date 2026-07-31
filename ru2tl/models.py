"""Структуры данных, которые ходят между стадиями пайплайна."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Segment:
    """Одна реплика: таймкод + исходный текст + перевод."""

    index: int
    start: float          # секунды от начала дорожки
    end: float
    text: str             # русский (ASR)
    translation: str = "" # тагальский (MT)
    audio: bytes = b""    # синтезированный PCM16 mono (TTS), сырой без заголовка

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class PipelineResult:
    segments: list[Segment] = field(default_factory=list)
    sample_rate: int = 24000

    @property
    def source_text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    @property
    def target_text(self) -> str:
        return " ".join(s.translation.strip() for s in self.segments if s.translation.strip())
