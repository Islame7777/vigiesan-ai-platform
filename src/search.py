import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
from sentence_transformers import SentenceTransformer, CrossEncoder

from config import (QDRANT_URL, TEXT_COLLECTION, IMAGE_COLLECTION,
                    EMBEDDING_MODEL, CLIP_MODEL, RERANKER_MODEL)

client = QdrantClient(url=QDRANT_URL, timeout=60)
bge = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
clip = SentenceTransformer(CLIP_MODEL, device="cpu")
reranker = CrossEncoder(RERANKER_MODEL, device="cpu", max_length=512)

# Le jeu de questions (page 7) ne sert jamais de preuve
NOT_EVAL = FieldCondition(key="type", match=MatchValue(value="eval_question"))


def _filter(groups, exclude_eval):
    """Filtre applique DANS Qdrant, avant toute lecture des fiches.
    groups=None : pas de filtre de groupe (usage dev/test uniquement,
    l'agent passe toujours les groupes de l'utilisateur)."""
    must = []
    if groups is not None:
        must.append(FieldCondition(key="access_groups",
                                   match=MatchAny(any=list(groups))))
    must_not = [NOT_EVAL] if exclude_eval else None
    if not must and not must_not:
        return None
    return Filter(must=must or None, must_not=must_not)


def search_text(q, k=10, groups=None):
    if groups is not None and len(groups) == 0:
        return []
    v = bge.encode(q, normalize_embeddings=True).tolist()
    res = client.query_points(TEXT_COLLECTION, query=v, limit=k,
                              with_payload=True,
                              query_filter=_filter(groups, True))
    return [p.payload for p in res.points]


def search_images(q, k=3, groups=None):
    if groups is not None and len(groups) == 0:
        return []
    v = clip.encode(q, normalize_embeddings=True).tolist()
    res = client.query_points(IMAGE_COLLECTION, query=v, limit=k,
                              with_payload=True,
                              query_filter=_filter(groups, False))
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
        "Où intervient le zoning ?",
    ]
    for q in questions:
        print(f"\n=== {q}")
        candidats = search_text(q, 10, groups=["technicien"])

        print("  -- Qdrant seul (avant reranker)")
        for d in candidats[:3]:
            print(f"     p.{d['page']} | {d['element_id']} | {d['text'][:70]!r}")

        print("  -- Après reranker")
        for d in rerank(q, candidats, 3):
            print(f"     {d['rerank_score']:.2f} | p.{d['page']} | "
                  f"{d['element_id']} | {d['text'][:70]!r}")

    print("\n=== Recherche visuelle : 'orange warning light'")
    for d in search_images("orange warning light", groups=["technicien"]):
        print(f"     p.{d['page']} | {d['element_id']}")
