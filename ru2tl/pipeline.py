"""Общее ядро пайплайна: ASR → MT → TTS.

Обе обёртки — CLI (файл→файл) и стриминг с микрофона — дёргают этот класс.
Разница между ними только в том, откуда берётся аудио и как далеко нужно
смотреть по контексту; сами стадии одни и те же.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .audio import assemble_dub, encode_from_pcm16, format_timestamp
from .config import Config
from .models import PipelineResult, Segment
from .providers import ASR, MT, TTS, make_asr, make_mt, make_tts

ProgressFn = Callable[[str], None]


def _noop(_: str) -> None:
    pass


class SpeechPipeline:
    def __init__(
        self,
        cfg: Config,
        asr: ASR | None = None,
        mt: MT | None = None,
        tts: TTS | None = None,
        on_progress: ProgressFn = _noop,
    ):
        self.cfg = cfg
        self.log = on_progress
        # Провайдеры создаются лениво: режим «только текст» не должен требовать
        # ключей от TTS, которые ему не нужны.
        self._asr, self._mt, self._tts = asr, mt, tts

    @property
    def asr(self) -> ASR:
        if self._asr is None:
            self._asr = make_asr(self.cfg)
        return self._asr

    @property
    def mt(self) -> MT:
        if self._mt is None:
            self._mt = make_mt(self.cfg)
        return self._mt

    @property
    def tts(self) -> TTS:
        if self._tts is None:
            self._tts = make_tts(self.cfg)
        return self._tts

    # ------------------------------------------------------------------
    # Батч: файл → файл
    # ------------------------------------------------------------------

    def run_file(
        self,
        src: str | Path,
        dest: str | Path | None = None,
        write_subtitles: bool = True,
        synthesize: bool = True,
    ) -> PipelineResult:
        """synthesize=False — только распознавание и перевод (TTS не нужен)."""
        src = Path(src)
        total = 3 if synthesize else 2

        self.log(f"[1/{total}] распознавание ({self.cfg.asr_model})…")
        segments = self.asr.transcribe_file(str(src))
        if not segments:
            raise RuntimeError(f"в {src} не найдено речи")
        self.log(f"      сегментов: {len(segments)}")

        self.log(f"[2/{total}] перевод ru → {self.cfg.target_lang} ({self.cfg.mt_model})…")
        self.mt.translate_batch(segments)

        sample_rate = self.cfg.tts_sample_rate
        if synthesize:
            self.log(f"[3/3] синтез ({self.cfg.tts_provider}, {self.cfg.tts_voice})…")
            for seg in segments:
                seg.audio = self.tts.synthesize(seg.translation).tobytes()
            sample_rate = self.tts.sample_rate

        result = PipelineResult(segments=segments, sample_rate=sample_rate)

        if synthesize and dest:
            self._render(result, Path(dest))
            self.log(f"      готово: {dest}")

        if write_subtitles:
            base = Path(dest).with_suffix("") if dest else src.with_suffix("")
            self.write_srt(result, f"{base}.tl.srt", target=True)
            self.write_srt(result, f"{base}.ru.srt", target=False)
            self.log(f"      субтитры: {base}.ru.srt, {base}.tl.srt")

        return result

    def _render(self, result: PipelineResult, dest: Path) -> None:
        clips = [
            (seg.start, np.frombuffer(seg.audio, dtype=np.int16))
            for seg in result.segments
            if seg.audio
        ]
        total = max((s.end for s in result.segments), default=0.0)
        track = assemble_dub(
            clips,
            result.sample_rate,
            total,
            fit_timing=self.cfg.fit_timing,
            max_tempo=self.cfg.fit_max_tempo,
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        encode_from_pcm16(track, result.sample_rate, dest)

    @staticmethod
    def write_srt(result: PipelineResult, path: str | Path, target: bool = True) -> None:
        lines: list[str] = []
        for i, seg in enumerate(result.segments, start=1):
            text = seg.translation if target else seg.text
            if not text.strip():
                continue
            lines.append(str(i))
            lines.append(f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}")
            lines.append(text.strip())
            lines.append("")
        Path(path).write_text("\n".join(lines), encoding="utf-8")

    # ------------------------------------------------------------------
    # Стриминг: реплика → реплика
    # ------------------------------------------------------------------

    def run_utterance(
        self,
        pcm: np.ndarray,
        sample_rate: int,
        context: list[tuple[str, str]] | None = None,
        synthesize: bool = True,
    ) -> tuple[str, str, np.ndarray]:
        """Один кусок русской речи → (ru-текст, tl-текст, tl-аудио).

        context — предыдущие пары (ru, tl); обновляется вызывающей стороной.
        synthesize=False — вернуть только текст, TTS не трогать.
        """
        silence = np.zeros(0, dtype=np.int16)

        ru = self.asr.transcribe_pcm(pcm, sample_rate)
        if not ru:
            return "", "", silence

        tl = self.mt.translate_one(ru, context or [])
        if not tl or not synthesize:
            return ru, tl, silence

        return ru, tl, self.tts.synthesize(tl)

    def run_stream(
        self,
        utterances: Iterable[np.ndarray],
        sample_rate: int,
        synthesize: bool = True,
    ) -> Iterable[tuple[str, str, np.ndarray]]:
        """Обёртка над run_utterance, которая сама тащит скользящий контекст."""
        context: list[tuple[str, str]] = []
        for pcm in utterances:
            ru, tl, audio = self.run_utterance(pcm, sample_rate, context, synthesize)
            if ru and tl:
                context.append((ru, tl))
                context = context[-8:]
            yield ru, tl, audio
