"""Стриминговый режим: микрофон → тагальская речь в динамики, near-real-time.

Конвейер из трёх потоков, связанных очередями, чтобы захват не простаивал,
пока предыдущая реплика уходит в облако:

    микрофон ──► VAD-сегментатор ──► worker (ASR→MT→TTS) ──► воспроизведение
              (callback)      utterance_q                audio_q

Задержка — примерно длительность реплики + 1.5–4 с на три сетевых вызова.
Полностью убрать её нельзя: перевод требует законченной мысли, а тагальский
порядок слов не совпадает с русским, так что переводить «на лету» по словам
здесь нельзя в принципе.
"""

from __future__ import annotations

import queue
import sys
import threading
from dataclasses import dataclass

import numpy as np

from .config import Config
from .pipeline import SpeechPipeline

SAMPLE_RATE = 16000        # webrtcvad принимает 8/16/32/48 кГц
FRAME_MS = 30              # и кадры только по 10/20/30 мс
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000


@dataclass
class VadSettings:
    aggressiveness: int = 2      # 0 (мягко) … 3 (агрессивно режет шум)
    start_frames: int = 3        # сколько кадров речи подряд открывают реплику
    end_frames: int = 25         # ~750 мс тишины закрывают реплику
    max_utterance_s: float = 20.0  # аварийная отсечка для монолога без пауз
    min_utterance_s: float = 0.4   # короче — считаем щелчком, а не речью


class EnergyVad:
    """Запасной детектор речи на одном numpy — без C-расширений.

    Сравнивает RMS кадра с плавающим шумовым порогом: порог быстро
    подстраивается на тишине и почти не двигается на речи, поэтому детектор
    сам адаптируется к уровню фона. В шумной комнате webrtcvad точнее,
    но для нарезки реплик по паузам этого достаточно.
    """

    _RATIOS = {0: 2.0, 1: 2.5, 2: 3.0, 3: 4.0}

    def __init__(self, aggressiveness: int = 2):
        self.ratio = self._RATIOS.get(aggressiveness, 3.0)
        self.noise: float | None = None

    def is_speech(self, frame: bytes, _sample_rate: int) -> bool:
        x = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(x * x)) + 1e-6)
        if self.noise is None:
            self.noise = rms
        speech = rms > max(self.noise * self.ratio, 120.0)
        if speech:
            # На речи порог почти замораживаем, иначе он «догонит» голос.
            self.noise = 0.995 * self.noise + 0.005 * min(rms, self.noise * self.ratio)
        else:
            self.noise = 0.9 * self.noise + 0.1 * rms
        return speech


def make_vad(aggressiveness: int):
    """webrtcvad, если он установлен и импортируется; иначе — EnergyVad."""
    try:
        import webrtcvad  # noqa: PLC0415

        return webrtcvad.Vad(aggressiveness)
    except Exception:  # noqa: BLE001 — ModuleNotFoundError, а на 3.12+ ещё и pkg_resources
        return EnergyVad(aggressiveness)


class VadSegmenter:
    """Режет непрерывный поток кадров на реплики по паузам."""

    def __init__(self, settings: VadSettings | None = None):
        self.s = settings or VadSettings()
        self.vad = make_vad(self.s.aggressiveness)
        self._buf: list[np.ndarray] = []
        self._voiced = 0
        self._silence = 0
        self._active = False

    def push(self, frame: np.ndarray) -> np.ndarray | None:
        """Скормить один кадр. Вернуть готовую реплику, если она завершилась."""
        is_speech = self.vad.is_speech(frame.tobytes(), SAMPLE_RATE)

        if not self._active:
            self._voiced = self._voiced + 1 if is_speech else 0
            # Держим небольшой преролл, иначе съедается начало первого слова.
            self._buf.append(frame)
            self._buf = self._buf[-self.s.start_frames * 2 :]
            if self._voiced >= self.s.start_frames:
                self._active = True
                self._silence = 0
            return None

        self._buf.append(frame)
        self._silence = 0 if is_speech else self._silence + 1

        too_long = len(self._buf) * FRAME_MS / 1000 >= self.s.max_utterance_s
        if self._silence >= self.s.end_frames or too_long:
            return self._flush()
        return None

    def _flush(self) -> np.ndarray | None:
        pcm = np.concatenate(self._buf) if self._buf else np.zeros(0, dtype=np.int16)
        self._buf, self._voiced, self._silence, self._active = [], 0, 0, False
        if pcm.size / SAMPLE_RATE < self.s.min_utterance_s:
            return None
        return pcm

    def finish(self) -> np.ndarray | None:
        return self._flush() if self._active else None


def run_stream(
    cfg: Config,
    vad_settings: VadSettings | None = None,
    input_device: int | None = None,
    output_device: int | None = None,
    text_only: bool = False,
) -> None:
    """text_only=True — печатать перевод в терминал, не синтезируя речь.

    Полезно, чтобы проверить микрофон и качество перевода, пока не настроен TTS.
    """
    import sounddevice as sd

    pipeline = SpeechPipeline(cfg)
    segmenter = VadSegmenter(vad_settings)

    frame_q: queue.Queue[np.ndarray | None] = queue.Queue()
    utterance_q: queue.Queue[np.ndarray | None] = queue.Queue()
    audio_q: queue.Queue[np.ndarray | None] = queue.Queue()
    stop = threading.Event()

    def on_audio(indata, _frames, _time, status):
        if status:
            print(f"[mic] {status}", file=sys.stderr)
        frame_q.put(indata[:, 0].copy())

    def segment_loop() -> None:
        while not stop.is_set():
            frame = frame_q.get()
            if frame is None:
                break
            utt = segmenter.push(frame)
            if utt is not None:
                utterance_q.put(utt)
        tail = segmenter.finish()
        if tail is not None:
            utterance_q.put(tail)
        utterance_q.put(None)

    def worker_loop() -> None:
        context: list[tuple[str, str]] = []
        while True:
            utt = utterance_q.get()
            if utt is None:
                break
            try:
                ru, tl, audio = pipeline.run_utterance(
                    utt, SAMPLE_RATE, context, synthesize=not text_only
                )
            except Exception as exc:  # noqa: BLE001 — одна упавшая реплика не роняет сессию
                print(f"\n[ошибка] {exc}", file=sys.stderr)
                continue
            if not ru:
                continue
            print(f"\n  RU: {ru}\n  TL: {tl}", flush=True)
            if tl:
                context.append((ru, tl))
                context = context[-8:]
            if audio.size:
                audio_q.put(audio)
        audio_q.put(None)

    def playback_loop() -> None:
        with sd.OutputStream(
            samplerate=pipeline.tts.sample_rate, channels=1, dtype="int16",
            device=output_device,
        ) as out:
            while True:
                audio = audio_q.get()
                if audio is None:
                    break
                out.write(audio.reshape(-1, 1))

    threads = [
        threading.Thread(target=segment_loop, daemon=True),
        threading.Thread(target=worker_loop, daemon=True),
    ]
    if not text_only:
        # Поток воспроизведения трогает pipeline.tts, а он требует ключей —
        # в текстовом режиме его быть не должно.
        threads.append(threading.Thread(target=playback_loop, daemon=True))
    for t in threads:
        t.start()

    mode = "только текст, без озвучки" if text_only else "с озвучкой"
    device_name = sd.query_devices(input_device, "input")["name"] if input_device is not None \
        else sd.query_devices(kind="input")["name"]
    print(f"Слушаю: {device_name} ({mode}).")
    print(f"ru → {cfg.target_lang_name}. Ctrl+C — выход.\n")
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16",
            blocksize=FRAME_SAMPLES, device=input_device, callback=on_audio,
        ):
            while not stop.is_set():
                threading.Event().wait(0.2)
    except KeyboardInterrupt:
        print("\nЗавершаю…")
    finally:
        stop.set()
        frame_q.put(None)
        for t in threads:
            t.join(timeout=30)
