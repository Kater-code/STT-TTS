"""ASR через OpenAI Whisper API.

Батч: whisper-1 + verbose_json — единственная комбинация, дающая посегментные
таймкоды, которые нужны для дубляжа. Стриминг: gpt-4o-mini-transcribe,
таймкоды там не нужны, зато заметно ниже задержка.
"""

from __future__ import annotations

import numpy as np
from openai import OpenAI

from ..audio import decode_to_pcm16, pcm16_to_wav_bytes
from ..config import Config
from ..models import Segment

# Whisper принимает файлы до 25 МБ. 16 кГц PCM16 mono ≈ 32 КБ/с,
# значит один запрос безопасно вмещает ~10 минут.
_CHUNK_SECONDS = 600
_ASR_SAMPLE_RATE = 16000


class OpenAIWhisperASR:
    def __init__(self, cfg: Config, client: OpenAI | None = None):
        self.cfg = cfg
        self.client = client or OpenAI()

    # ---------- батч ----------

    def transcribe_file(self, path: str) -> list[Segment]:
        pcm = decode_to_pcm16(path, _ASR_SAMPLE_RATE)
        chunk = _CHUNK_SECONDS * _ASR_SAMPLE_RATE

        segments: list[Segment] = []
        for offset in range(0, pcm.size, chunk):
            piece = pcm[offset : offset + chunk]
            base = offset / _ASR_SAMPLE_RATE
            segments.extend(self._transcribe_chunk(piece, base, len(segments)))
        return segments

    def _transcribe_chunk(self, pcm: np.ndarray, time_offset: float, index_offset: int) -> list[Segment]:
        wav = pcm16_to_wav_bytes(pcm, _ASR_SAMPLE_RATE)
        resp = self.client.audio.transcriptions.create(
            model=self.cfg.asr_model,
            file=("audio.wav", wav, "audio/wav"),
            language=self.cfg.source_lang,
            prompt=self.cfg.asr_prompt or None,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

        raw = getattr(resp, "segments", None) or []
        if not raw:
            # Модель без таймкодов — оставляем один сегмент на весь кусок.
            text = (getattr(resp, "text", "") or "").strip()
            if not text:
                return []
            return [Segment(index=index_offset, start=time_offset,
                            end=time_offset + pcm.size / _ASR_SAMPLE_RATE, text=text)]

        out: list[Segment] = []
        for s in raw:
            text = (s.text if hasattr(s, "text") else s["text"]).strip()
            if not text:
                continue
            start = float(s.start if hasattr(s, "start") else s["start"])
            end = float(s.end if hasattr(s, "end") else s["end"])
            out.append(Segment(
                index=index_offset + len(out),
                start=time_offset + start,
                end=time_offset + end,
                text=text,
            ))
        return out

    # ---------- стриминг ----------

    def transcribe_pcm(self, pcm: np.ndarray, sample_rate: int) -> str:
        wav = pcm16_to_wav_bytes(pcm, sample_rate)
        resp = self.client.audio.transcriptions.create(
            model=self.cfg.asr_stream_model,
            file=("chunk.wav", wav, "audio/wav"),
            language=self.cfg.source_lang,
            prompt=self.cfg.asr_prompt or None,
            response_format="text",
        )
        return (resp if isinstance(resp, str) else getattr(resp, "text", "")).strip()
