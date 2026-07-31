"""Аудио-утилиты: декодирование через ffmpeg, WAV-обёртки, сборка дубляжа.

Единый внутренний формат — PCM16 mono, numpy int16.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np


class FfmpegMissing(RuntimeError):
    pass


def require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise FfmpegMissing(
            "ffmpeg не найден в PATH. macOS: brew install ffmpeg"
        )
    return path


def decode_to_pcm16(src: str | Path, sample_rate: int = 16000) -> np.ndarray:
    """Любой аудио/видео файл → PCM16 mono numpy-массив на заданной частоте."""
    ffmpeg = require_ffmpeg()
    proc = subprocess.run(
        [ffmpeg, "-nostdin", "-v", "error", "-i", str(src),
         "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(sample_rate), "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg не смог декодировать {src}: {proc.stderr.decode(errors='replace')}")
    return np.frombuffer(proc.stdout, dtype=np.int16)


def pcm16_to_wav_bytes(pcm: np.ndarray | bytes, sample_rate: int) -> bytes:
    """PCM16 → WAV-контейнер в памяти (то, что ждёт OpenAI audio API)."""
    raw = pcm.tobytes() if isinstance(pcm, np.ndarray) else pcm
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(raw)
    return buf.getvalue()


def write_wav(path: str | Path, pcm: np.ndarray, sample_rate: int) -> None:
    Path(path).write_bytes(pcm16_to_wav_bytes(pcm, sample_rate))


def encode_from_pcm16(pcm: np.ndarray, sample_rate: int, dest: str | Path) -> None:
    """PCM16 → любой формат, который поймёт ffmpeg по расширению (mp3/m4a/wav/...)."""
    ffmpeg = require_ffmpeg()
    proc = subprocess.run(
        [ffmpeg, "-nostdin", "-v", "error", "-y",
         "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "-", str(dest)],
        input=pcm.tobytes(), capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg не смог записать {dest}: {proc.stderr.decode(errors='replace')}")


def _atempo_chain(factor: float) -> str:
    """atempo принимает 0.5..2.0 за фильтр — большие коэффициенты набираем каскадом."""
    parts: list[str] = []
    remaining = factor
    while remaining > 2.0:
        parts.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        parts.append("atempo=0.5")
        remaining /= 0.5
    parts.append(f"atempo={remaining:.6f}")
    return ",".join(parts)


def time_stretch(pcm: np.ndarray, sample_rate: int, factor: float) -> np.ndarray:
    """Изменить темп без изменения высоты тона. factor > 1 — быстрее."""
    if abs(factor - 1.0) < 0.01:
        return pcm
    ffmpeg = require_ffmpeg()
    proc = subprocess.run(
        [ffmpeg, "-nostdin", "-v", "error",
         "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "-",
         "-filter:a", _atempo_chain(factor),
         "-f", "s16le", "-ac", "1", "-ar", str(sample_rate), "-"],
        input=pcm.tobytes(), capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg atempo упал: {proc.stderr.decode(errors='replace')}")
    return np.frombuffer(proc.stdout, dtype=np.int16)


def assemble_dub(
    clips: list[tuple[float, np.ndarray]],
    sample_rate: int,
    total_duration: float,
    fit_timing: bool = True,
    max_tempo: float = 1.35,
) -> np.ndarray:
    """Разложить синтезированные реплики по таймлайну оригинала.

    clips — [(start_sec, pcm), ...] в порядке возрастания start.
    Тагальский обычно длиннее русского на 10-25%, поэтому реплики норовят
    наехать друг на друга. Стратегия: сначала пробуем ускорить до max_tempo,
    чтобы влезть в исходное окно; если и так не влезает — сдвигаем следующую
    реплику вправо (накапливаем дрейф), но никогда не режем речь.
    """
    out = np.zeros(int(total_duration * sample_rate) + sample_rate, dtype=np.int16)
    cursor = 0  # сэмплы: раньше этой точки писать нельзя, там уже речь

    for i, (start, pcm) in enumerate(clips):
        if pcm.size == 0:
            continue
        pos = max(int(start * sample_rate), cursor)

        if fit_timing:
            next_start = clips[i + 1][0] if i + 1 < len(clips) else total_duration
            window = int(next_start * sample_rate) - pos
            if 0 < window < pcm.size:
                needed = pcm.size / window
                pcm = time_stretch(pcm, sample_rate, min(needed, max_tempo))

        end = pos + pcm.size
        if end > out.size:
            out = np.concatenate([out, np.zeros(end - out.size + sample_rate, dtype=np.int16)])
        # int32 при сложении, чтобы не словить переполнение на наложениях
        out[pos:end] = np.clip(out[pos:end].astype(np.int32) + pcm.astype(np.int32),
                               -32768, 32767).astype(np.int16)
        cursor = end

    return out[: max(cursor, int(total_duration * sample_rate))]


def format_timestamp(seconds: float) -> str:
    """SRT-таймкод: 00:01:23,456"""
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
