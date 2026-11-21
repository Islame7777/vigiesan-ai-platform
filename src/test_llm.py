import time

from agent import graph

TESTS = [
    "Que signifie AL-01 ?",
    "Quelle alerte touche la capacité ?",
    "Qui doit approuver la création du ticket ?",
]

for q in TESTS:
    t0 = time.time()
    out = graph.invoke({"question": q, "user_groups": ["technicien"]})
    a = out["answer"]
    print(f"\n=== {q}   ({time.time() - t0:.0f} s)")
    print("  statut      :", a["status"])
    print("  réponse     :", a["answer"])
    print("  citations   :", [(c["element_id"], c["page"]) for c in a["citations"]])
    print("  incertitudes:", a.get("uncertainties"))
    print("  verdict     :", out.get("verdict"))
