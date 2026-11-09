# ru2tl: Russian speech in, Tagalog speech out

A three-stage dubbing pipeline with one shared core and two entry points: file-to-file batch and live microphone streaming.

```
audio (ru) ──► ASR ──► MT ──► TTS ──► audio (tl)
             Whisper  GPT   Google fil-PH
```

| Stage | Default provider | Why |
|---|---|---|
| ASR | OpenAI `whisper-1` | Russian is one of Whisper's strongest locales; `verbose_json` returns per-segment timestamps needed for dubbing |
| MT | OpenAI `gpt-4o` | batched translation with shared context and a glossary |
| TTS | Google Cloud `fil-PH` | most reliable Tagalog coverage among cloud TTS (Standard and WaveNet voices) |

ElevenLabs `eleven_multilingual_v2` is available as an alternative TTS when the source speaker's voice should be cloned.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg
cp .env.example .env
```

Needs `OPENAI_API_KEY` and `GOOGLE_APPLICATION_CREDENTIALS` (service account with Cloud Text-to-Speech enabled).

## Usage

```bash
python -m ru2tl file input.mp4 output.wav --glossary glossary.example.json
python -m ru2tl stream --vad 2 --silence-ms 750
python -m ru2tl voices
```

Batch mode also writes `output.ru.srt` and `output.tl.srt` next to the output.

## Design notes

- `ru2tl/pipeline.py` is the only place where stages are connected. Both entry points call the same `SpeechPipeline`; providers sit behind three protocols in `providers/base.py`, so swapping Whisper for another ASR is one new file.
- Lines are translated in batches of 40 with numbering and a strict JSON schema, so mapping `id -> translation` does not depend on model formatting. Batching keeps gender, number and references consistent; the last three pairs are carried across batch boundaries.
- The system prompt (style plus glossary) is byte-identical across requests, so it is eligible for OpenAI prompt caching.
- Streaming skips the schema to save latency and uses a sliding context of the last three pairs instead.
