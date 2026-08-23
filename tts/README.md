# TTS

MMS-TTS (VITS), two small (~139MB each) checkpoints instead of a single large model.

- **Hindi**: `facebook/mms-tts-hin`, native, sounds correct.
- **Khasi**: `facebook/mms-tts-vie` (Vietnamese) used as a proxy. Khasi and Vietnamese are
  both Austroasiatic (Mon-Khmer) languages, so this is a linguistically justified phoneme
  match, not an arbitrary Latin-script substitution. Output is functional but audibly
  robotic; a proper Khasi voice fine-tune on existing studio audio data is the planned
  next step before any in-person demo round.

## Why MMS-TTS over the original choice

The original TTS (OmniVoice) loaded an LLM backbone plus the HiggsAudioV2 audio codec
(~2.3GB + 770MB combined) onto the GPU alongside ASR, which repeatedly overflowed the
Jetson Orin Nano's 256MB CMA memory pool and crashed. MMS-TTS's much smaller footprint
fixed this completely; the pipeline now loads clean on the first attempt.

Proxy candidates tested before landing on Vietnamese: English, Welsh, Garo (all worked
technically but weaker linguistic justification or worse pronunciation).
