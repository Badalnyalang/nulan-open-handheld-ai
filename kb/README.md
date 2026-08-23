# Knowledge Base

- `ncert_kb_multilingual.json`: source KB, 510 entries, 210 NCERT-sourced topics
  (std 4-6 general science). Each entry has multiple question phrasings per language
  and answers in Khasi and Hindi, sourced directly from NCERT-based content (not
  machine-translated).
- `kb_answers.json`: flat lookup of `{entry_id: {"kha": ..., "hi": ...}}`, used by
  `retrieval/ne_embed.py` at runtime.
- `kb_index.npz`: precomputed embeddings for every question phrasing, built by
  `retrieval/build_kb_index.py`. Regenerate this after any change to the source KB.

NCERT content has a multi-year revision cycle, so the KB stays relevant without
frequent re-curation.

Place the actual data files here (not included in this skeleton).
