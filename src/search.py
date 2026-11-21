import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer, CrossEncoder

from config import (QDRANT_URL, TEXT_COLLECTION, IMAGE_COLLECTION,
                    EMBEDDING_MODEL, CLIP_MODEL, RERANKER_MODEL)

client = QdrantClient(url=QDRANT_URL, timeout=60)
bge = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
clip = SentenceTransformer(CLIP_MODEL, device="cpu")
reranker = CrossEncoder(RERANKER_MODEL, device="cpu", max_length=512)

# Le jeu de questions (page 7) ne sert jamais de preuve
NO_EVAL = Filter(must_not=[
    FieldCondition(key="type", match=MatchValue(value="eval_question"))])


def search_text(q, k=10):
    v = bge.encode(q, normalize_embeddings=True).tolist()
    res = client.query_points(TEXT_COLLECTION, query=v, limit=k,
                              with_payload=True, query_filter=NO_EVAL)
    return [p.payload for p in res.points]


def search_images(q, k=3):
    v = clip.encode(q, normalize_embeddings=True).tolist()
    res = client.query_points(IMAGE_COLLECTION, query=v, limit=k,
                              with_payload=True)
    return [p.payload for p in res.points]


def rerank(q, docs, top=3):
    scores = reranker.predict([(q, d["text"]) for d in docs])
    order = np.argsort(-scores)[:top]
    return [{**docs[i], "rerank_score": float(scores[i])} for i in order]


if __name__ == "__main__":
    questions = [
        "Que signifie AL-01 ?",
        "Quelle alerte touche la capacité ?",
        "Qui doit approuver la création du ticket ?",
    ]
    for q in questions:
        print(f"\n=== {q}")
        candidats = search_text(q, 10)

        print("  -- Qdrant seul (avant reranker)")
        for d in candidats[:3]:
            print(f"     p.{d['page']} | {d['element_id']} | {d['text'][:70]!r}")

        print("  -- Après reranker")
        for d in rerank(q, candidats, 3):
            print(f"     {d['rerank_score']:.2f} | p.{d['page']} | "
                  f"{d['element_id']} | {d['text'][:70]!r}")

    print("\n=== Recherche visuelle : 'orange warning light'")
    for d in search_images("orange warning light"):
        print(f"     p.{d['page']} | {d['element_id']}")
