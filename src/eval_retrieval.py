"""Mesure de la recherche sur les questions du manuel (Q01 a Q12)."""
import json
import math
import sys
import time
from pathlib import Path

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

import search
from config import TEXT_COLLECTION

GROUPS = ["technicien"]

# JUGEMENT HUMAIN : sources acceptables pour chaque question (ids de fiches).
GOLD = {
    "Q01": ("Quel est le role des deux fabrics ?",
            {"P02-TEXT", "P02-TAB01-L03", "P03-TEXT", "FIG-01-OCR", "FIG-02-OCR"}),
    "Q02": ("Ou intervient le zoning ?",
            {"P03-TAB01-L03", "P03-TEXT", "FIG-02-OCR"}),
    "Q03": ("Que signifie MPIO ?",
            {"P03-TAB01-L05", "P03-TEXT"}),
    "Q04": ("Le disque B03 a-t-il un antecedent ?",
            {"P07-TAB01-L02", "P07-TEXT", "FIG-03-OCR"}),
    "Q05": ("Quel est le sens de AL-01 ?",
            {"P05-TAB01-L01"}),
    "Q06": ("Le voyant orange permet-il un diagnostic certain ?",
            {"P05-TEXT"}),
    "Q07": ("Que faut-il verifier avant un ticket ?",
            {"P06-TEXT"}),
    "Q08": ("Qui doit approuver la creation du ticket ?",
            {"P06-TAB01-L03", "P06-TEXT"}),
    "Q09": ("Quelle procedure correspond a une perte de chemin ?",
            {"P03-TEXT"}),
    "Q10": ("Peut-on declencher un failover depuis MaintiDoc ?",
            {"P06-TAB01-L04", "P06-TEXT", "FIG-05-OCR"}),
    "Q11": ("Quelle alerte touche la capacite ?",
            {"P05-TAB01-L04"}),
    "Q12": ("Quels elements retrouve-t-on dans le ticket exemple ?",
            {"P07-TAB03-L01", "P07-TAB03-L02", "P07-TAB03-L03",
             "P07-TAB03-L04", "P07-TAB03-L05", "P07-TEXT"}),
}

CONFIGS = [
    ("A. Qdrant seul, sans OCR", False, False),
    ("B. Qdrant seul, avec OCR", True, False),
    ("C. + reranker, sans OCR", False, True),
    ("D. + reranker, avec OCR", True, True),
]

# 0) Les ids de reference existent-ils vraiment dans la base ?
pts, _ = search.client.scroll(TEXT_COLLECTION, limit=200, with_payload=True)
existing = {p.payload["element_id"] for p in pts}
missing = sorted({g for _, gs in GOLD.values() for g in gs} - existing)
if missing:
    print("Identifiants de reference absents de la base :", missing)
    sys.exit(1)
print(f"Base : {len(existing)} fiches ; references verifiees.")


def dense(q, k, with_ocr):
    must_not = [FieldCondition(key="type", match=MatchValue(value="eval_question"))]
    if not with_ocr:
        must_not.append(FieldCondition(key="type", match=MatchValue(value="schema_ocr")))
    flt = Filter(
        must=[FieldCondition(key="access_groups", match=MatchAny(any=GROUPS))],
        must_not=must_not)
    v = search.bge.encode(q, normalize_embeddings=True).tolist()
    res = search.client.query_points(TEXT_COLLECTION, query=v, limit=k,
                                     with_payload=True, query_filter=flt)
    return [p.payload for p in res.points]


def metrics(ranked_ids, gold):
    top5 = ranked_ids[:5]
    hits = [i for i in top5 if i in gold]
    hit = 1.0 if hits else 0.0
    recall = len(set(hits)) / len(gold)
    dcg = sum(1.0 / math.log2(r + 2) for r, i in enumerate(top5) if i in gold)
    idcg = sum(1.0 / math.log2(r + 2) for r in range(min(len(gold), 5)))
    ndcg = dcg / idcg
    rr, first = 0.0, None
    for r, i in enumerate(ranked_ids[:10], 1):
        if i in gold:
            rr, first = 1.0 / r, r
            break
    return hit, recall, ndcg, rr, first


t0 = time.time()
cache = {}
table = {}
for name, with_ocr, use_rr in CONFIGS:
    rows = []
    print(f"\n=== {name}")
    for qid, (q, gold) in GOLD.items():
        key = (qid, with_ocr)
        if key not in cache:
            cache[key] = dense(q, 10, with_ocr)
        cands = cache[key]
        if use_rr:
            ranked = search.rerank(q, cands, top=len(cands))
        else:
            ranked = cands
        ids = [d["element_id"] for d in ranked]
        m = metrics(ids, gold)
        rows.append((qid,) + m)
        print(f"  {qid}  1er bon rang : {str(m[4]):>4}   top3 : {ids[:3]}")
    table[name] = rows

print("\n" + "=" * 78)
print(f"{'Reglage':28} {'Reussite@5':>11} {'Recall@5':>9} {'nDCG@5':>8} {'MRR@10':>8}")
summary = {}
for name, rows in table.items():
    n = len(rows)
    avg = [sum(r[i] for r in rows) / n for i in (1, 2, 3, 4)]
    summary[name] = {"hit5": avg[0], "recall5": avg[1], "ndcg5": avg[2], "mrr10": avg[3]}
    print(f"{name:28} {avg[0]:>11.2f} {avg[1]:>9.2f} {avg[2]:>8.2f} {avg[3]:>8.2f}")

out = {
    "date": time.strftime("%Y-%m-%d"),
    "n_questions": len(GOLD),
    "summary": summary,
    "first_gold_rank": {name: {r[0]: r[5] for r in rows} for name, rows in table.items()},
    "note": ("Sources de reference = jugement humain. 12 questions seulement. "
             "Q11 contaminee (titres de tableaux ajoutes apres un echec)."),
}
dest = Path(__file__).resolve().parent.parent / "tests"
dest.mkdir(exist_ok=True)
(dest / "eval_baseline.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nDuree : {time.time() - t0:.0f} s. Resultats : tests/eval_baseline.json")
print("Rappel : 12 questions, reference = jugement humain, Q11 contaminee.")
