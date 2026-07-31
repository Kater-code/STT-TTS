"""CLI: python -m ru2tl file <in> <out>  |  python -m ru2tl stream  |  ... voices"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Config


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--glossary", help="JSON с глоссарием / style_guide / asr_prompt")
    p.add_argument("--mt-model", help="модель перевода (по умолчанию gpt-4o)")
    p.add_argument("--asr-model", help="модель распознавания (по умолчанию whisper-1)")
    p.add_argument("--tts-provider", choices=["google", "elevenlabs"])
    p.add_argument("--voice", help="голос TTS, напр. fil-PH-Wavenet-B")


def _build_config(args: argparse.Namespace) -> Config:
    overrides: dict = {}
    if getattr(args, "mt_model", None):
        overrides["mt_model"] = args.mt_model
    if getattr(args, "asr_model", None):
        overrides["asr_model"] = args.asr_model
    if getattr(args, "tts_provider", None):
        overrides["tts_provider"] = args.tts_provider
    if getattr(args, "voice", None):
        overrides["tts_voice"] = args.voice
    if getattr(args, "no_fit_timing", False):
        overrides["fit_timing"] = False
    return Config.load(getattr(args, "glossary", None), **overrides)


def main(argv: list[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(
        prog="ru2tl",
        description="Русская речь на входе → тагальская речь на выходе.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_file = sub.add_parser("file", help="батч: аудио/видео файл → озвученная дорожка")
    p_file.add_argument("input", help="любой формат, который читает ffmpeg")
    p_file.add_argument("output", nargs="?",
                        help="выходное аудио (.wav / .mp3 / .m4a); не нужно с --text-only")
    p_file.add_argument("--text-only", action="store_true",
                        help="только распознать и перевести, без синтеза (не требует TTS-ключей)")
    p_file.add_argument("--no-subtitles", action="store_true", help="не писать .srt")
    p_file.add_argument("--no-fit-timing", action="store_true",
                        help="не подгонять реплики под тайминг оригинала")
    _add_common(p_file)

    p_stream = sub.add_parser("stream", help="near-real-time: микрофон → динамики")
    p_stream.add_argument("--input-device", type=int)
    p_stream.add_argument("--output-device", type=int)
    p_stream.add_argument("--vad", type=int, default=2, choices=[0, 1, 2, 3],
                          help="агрессивность VAD: 0 мягко … 3 жёстко (по умолчанию 2)")
    p_stream.add_argument("--silence-ms", type=int, default=750,
                          help="пауза, закрывающая реплику (по умолчанию 750)")
    p_stream.add_argument("--text-only", action="store_true",
                          help="печатать перевод в терминал без озвучки (не требует TTS-ключей)")
    _add_common(p_stream)

    p_voices = sub.add_parser("voices", help="показать доступные голоса fil-PH")
    _add_common(p_voices)

    p_devices = sub.add_parser("devices", help="показать аудиоустройства")

    args = parser.parse_args(argv)

    if args.command == "devices":
        import sounddevice as sd

        print(sd.query_devices())
        return 0

    cfg = _build_config(args)

    if args.command == "voices":
        from .providers.tts_google import GoogleTTS

        for name in GoogleTTS(cfg).list_voices():
            print(name)
        return 0

    if args.command == "file":
        from .pipeline import SpeechPipeline

        src = Path(args.input)
        if not src.exists():
            print(f"нет такого файла: {src}", file=sys.stderr)
            return 1
        if not args.output and not args.text_only:
            print("укажите выходной файл или добавьте --text-only", file=sys.stderr)
            return 1

        pipeline = SpeechPipeline(cfg, on_progress=lambda m: print(m, flush=True))
        result = pipeline.run_file(
            src, args.output,
            write_subtitles=not args.no_subtitles,
            synthesize=not args.text_only,
        )

        if args.text_only:
            print()
            for seg in result.segments:
                print(f"[{seg.start:7.2f}] RU: {seg.text}")
                print(f"{'':10}TL: {seg.translation}\n")
        print(f"сегментов: {len(result.segments)}")
        return 0

    if args.command == "stream":
        from .stream import FRAME_MS, VadSettings, run_stream

        run_stream(
            cfg,
            VadSettings(
                aggressiveness=args.vad,
                end_frames=max(1, args.silence_ms // FRAME_MS),
            ),
            input_device=args.input_device,
            output_device=args.output_device,
            text_only=args.text_only,
        )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
