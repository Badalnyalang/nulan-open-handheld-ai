# ASR

Own Whisper fine-tune, plus a dedicated Hindi conformer path.

- **Khasi**: transcribed via `ASR.transcribe_khasi()`, using a Welsh (`cy`) language token as a
  proxy since Whisper has no native Khasi support. ~14% WER on the fine-tuned model.
- **Hindi**: handled separately by `HindiASR`, a dedicated ONNX conformer + CTC decoder bundled
  as `hi-conformer.onnx`, giving accurate Devanagari output. Not routed through Whisper's own
  Hindi token, which degrades under the Khasi-focused fine-tune.
- **English**: not used in the current demo scope (Khasi + Hindi only).

Model weights: `MWirelabs/ne-asr` on Hugging Face.

Warm inference: ASR is the dominant latency cost in the pipeline (~2.3-2.5s), see
`pipeline/latency_bench.py` for the full breakdown.
