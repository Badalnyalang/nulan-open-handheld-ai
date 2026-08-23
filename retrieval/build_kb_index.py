"""
One-time offline job: embeds every question phrasing in kb_answers.json
and writes a flat index (kb_index.npz) for fast similarity search at runtime.

Run once after any KB content change. Not part of the live pipeline.
"""
import json
import numpy as np
from sentence_transformers import SentenceTransformer


def build_index(kb_source_path, embed_model_dir, output_path):
    with open(kb_source_path, encoding="utf-8") as f:
        kb = json.load(f)

    model = SentenceTransformer(embed_model_dir, device="cpu")

    ids, texts = [], []
    for entry in kb:
        entry_id = entry["id"]
        for lang, phrasings in entry["questions"].items():
            for phrasing in phrasings:
                ids.append(entry_id)
                texts.append(phrasing)

    print(f"Embedding {len(texts)} question phrasings across {len(kb)} KB entries...")
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    np.savez(output_path, ids=np.array(ids), vectors=np.array(vectors))
    print(f"Saved index to {output_path}")


if __name__ == "__main__":
    build_index(
        kb_source_path="../kb/ncert_kb_multilingual.json",
        embed_model_dir="./ne-embed",
        output_path="../kb/kb_index.npz",
    )
