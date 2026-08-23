# Retrieval

`ne-embed`, a LaBSE-based cross-lingual sentence embedding model, matches the transcribed
query against a precomputed index of every question phrasing in the KB.

- `build_kb_index.py`: offline, one-time. Embeds all question phrasings from the source
  KB (`ncert_kb_multilingual.json`) and writes `kb/kb_index.npz`. Re-run only when KB
  content changes.
- `ne_embed.py`: runtime retrieval. Embeds the incoming query and does a single similarity
  search against the cached index, CPU-only.

No live translation happens here, Khasi and Hindi answers are sourced directly from the
NCERT-based content, not machine-translated at query time.

Warm latency: ~0.68-1.87s depending on run (see `pipeline/latency_bench.py`).
