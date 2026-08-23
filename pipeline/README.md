# Pipeline

`run_pipeline.py`: loads all four models (ASR, LID, retrieval, TTS) once, then runs
mic-to-speaker end to end for a single query.

Flow: audio in -> ASR (Khasi transcription) -> LID (Hindi vs Khasi routing) -> retrieval
(KB match) -> TTS (spoken answer in the routed language) -> audio out.

`latency_bench.py`: runs the same query twice in one process (cold load, then warm) to
measure realistic per-stage and total latency. This produced the validated numbers in the
architecture doc:

| Stage | Run 1 | Run 2 (warm) |
|---|---|---|
| ASR | 2.32s | 2.45s |
| LID | ~0s | ~0s |
| Retrieval | 1.87s | 0.68s |
| TTS | 0.72s | 0.50s |
| Total | 4.9s | 3.6s |

ASR is the dominant cost, not retrieval or TTS as originally assumed.
