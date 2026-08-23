import json
import numpy as np
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, embed_model_dir, kb_index_path, kb_answers_path, device="cpu"):
        self.model = SentenceTransformer(embed_model_dir, device=device)
        index = np.load(kb_index_path, allow_pickle=True)
        self.ids = index["ids"]
        self.vectors = index["vectors"]
        with open(kb_answers_path, encoding="utf-8") as f:
            self.answers = json.load(f)

    def match(self, query_text, top_k=1):
        query_vec = self.model.encode([query_text], normalize_embeddings=True)[0]
        scores = self.vectors @ query_vec
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_idx:
            entry_id = self.ids[idx]
            results.append({
                "id": entry_id,
                "score": float(scores[idx]),
                "answers": self.answers.get(entry_id, {}),
            })
        return results


if __name__ == "__main__":
    r = Retriever(
        embed_model_dir="./ne-embed",
        kb_index_path="../kb/kb_index.npz",
        kb_answers_path="../kb/kb_answers.json",
    )
    for match in r.match("What is photosynthesis?"):
        print(match)
