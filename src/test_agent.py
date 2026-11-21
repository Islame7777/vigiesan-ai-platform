from agent import graph

TESTS = [
    ("Que signifie AL-01 ?", ["technicien"]),
    ("Quelle alerte touche la capacité ?", ["technicien"]),
    ("Le voyant orange est allumé, quelle est la cause ?", ["technicien"]),
    ("Quelle est la recette de la pizza ?", ["technicien"]),
    ("Que signifie AL-01 ?", []),
]

for q, groups in TESTS:
    out = graph.invoke({"question": q, "user_groups": groups})
    a = out["answer"]
    print(f"\n=== {q}  (groupes={groups})")
    print("  statut     :", a["status"])
    print("  réponse    :", a["answer"][:90])
    if a.get("clarification_question"):
        print("  question   :", a["clarification_question"][:90])
    print("  citations  :", [c["element_id"] for c in a.get("citations", [])])
    print("  recherches :", out.get("searches", 0), "| verdict :", out.get("verdict"))