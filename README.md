# Suno Sutra

Offline, voice-first, multilingual education assistant for the VYOMA Innovation Challenge.
Khasi + Hindi general science Q&A for std 4-6, running fully on-device on a Jetson Orin Nano.

## Pipeline

```
USB Mic -> ASR (GPU) -> LID (CPU) -> ne-embed retrieval (CPU) -> KB Match -> TTS (GPU) -> Speaker
```

- **ASR**: own Whisper fine-tune. Khasi routed through a Welsh proxy language token (~14% WER), Hindi via a dedicated conformer path.
- **LID**: IndicLID (AI4Bharat, fastText, CPU-only). Detects Hindi and routes away from the Khasi default.
- **Retrieval**: `ne-embed`, LaBSE-based cross-lingual embedding match against a precomputed KB index.
- **KB**: 510-entry Khasi/Hindi knowledge base, 210 NCERT-sourced topics (std 4-6 general science). Built and cached offline, no live translation or generation at runtime.
- **TTS**: MMS-TTS (VITS), two small (~139MB) checkpoints: `facebook/mms-tts-hin` (Hindi, native) and `facebook/mms-tts-vie` (Vietnamese, used as the Khasi proxy since Khasi and Vietnamese are both Austroasiatic/Mon-Khmer languages).

## Why this architecture

The original TTS choice (OmniVoice) loaded an LLM backbone plus the HiggsAudioV2 audio codec (~2.3GB + 770MB combined) alongside ASR on the Jetson's 8GB unified memory, which repeatedly crashed the GPU's NvMap/CMA allocator (only a 256MB pool). Swapping to MMS-TTS's much smaller VITS checkpoints fixed this completely, the full pipeline now loads clean on first attempt with no retries.

## Validated latency

| Stage | Run 1 | Run 2 (warm) |
|---|---|---|
| ASR | 2.32s | 2.45s |
| LID | ~0s | ~0s |
| Retrieval | 1.87s | 0.68s |
| TTS | 0.72s | 0.50s |
| **Total** | **4.9s** | **3.6s** |

Target budget: 5-8s.

## Repo layout

```
asr/          Whisper-based ASR wrapper (Khasi + Hindi)
lid/          Language ID / routing
retrieval/    KB embedding + similarity search
tts/          MMS-TTS synthesis wrapper
pipeline/     End-to-end orchestration + latency benchmark
kb/           Knowledge base data and prebuilt index
hardware/     Jetson setup and device notes
docs/         Submission documents (architecture, business plan, BOM)
```

## Setup

```bash
pip install -r requirements.txt
python retrieval/build_kb_index.py     # one-time: builds kb/kb_index.npz
python pipeline/run_pipeline.py        # runs one query end-to-end
```

## Hardware

Jetson Orin Nano 8GB, JetPack 6.2.1. See `hardware/README.md`.
