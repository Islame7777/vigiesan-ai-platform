import re
from typing import TypedDict
from llm import ask_llm

from langgraph.graph import StateGraph, START, END

from search import search_text, rerank
from citations import Answer, Citation, check_citations

MAX_SEARCHES = 2      # limite de boucle : jamais plus de 2 recherches
MIN_SCORE = 0.05      # seuil de départ, à mesurer à l'étape 5
ALARM_CODE = re.compile(r"\bAL[- ]?\d{2}\b", re.IGNORECASE)
VOYANT = re.compile(r"\bvoyants?\b|\bLED\b", re.IGNORECASE)


class State(TypedDict, total=False):
    question: str
    user_groups: list
    evidence: list
    searches: int
    answer: dict
    verdict: str


# ---------- les cases du graphe ----------
def node_access(state):
    if not state.get("user_groups"):
        return {"answer": {"status": "refused",
                           "answer": "Accès refusé : aucun groupe d'accès.",
                           "citations": []},
                "verdict": "refusé"}
    return {}


def node_analyze(state):
    q = state["question"]
    if VOYANT.search(q) and not ALARM_CODE.search(q):
        return {"answer": {
            "status": "needs_clarification",
            "answer": "Un voyant seul ne permet pas d'identifier la cause.",
            "clarification_question": "Quel est le code d'alarme affiché, "
                                      "le modèle de la baie et le contexte du panneau ?",
            "citations": []},
            "verdict": "clarification demandée"}
    return {}


def node_search(state):
    n = state.get("searches", 0)
    k, top = (10, 3) if n == 0 else (20, 5)   # 2e essai : plus large
    ev = rerank(state["question"], search_text(state["question"], k), top)
    groups = set(state["user_groups"])
    ev = [e for e in ev if groups & set(e["access_groups"])]
    return {"evidence": ev, "searches": n + 1}


def node_clarify(state):
    return {"answer": {
        "status": "needs_clarification",
        "answer": "Je n'ai pas trouvé de source assez pertinente "
                  "dans les documents autorisés.",
        "clarification_question": "Pouvez-vous reformuler ou préciser "
                                  "l'équipement et le code d'alarme ?",
        "citations": []},
        "verdict": "clarification : preuves insuffisantes"}


def node_answer(state):
    ev = state["evidence"]
    try:
        out = ask_llm(state["question"], ev)
    except Exception as e:
        return {"answer": {
            "status": "needs_clarification",
            "answer": "Le modèle de langage est indisponible.",
            "citations": [],
            "uncertainties": [type(e).__name__]},
            "verdict": "erreur modèle"}

    by_id = {e["element_id"]: e for e in ev}
    cits = []
    for cid in out.cited_ids:
        e = by_id.get(cid)
        if e:
            cits.append(Citation(document_id=e["document_id"],
                                 element_id=cid, page=e["page"]))
        else:   # identifiant inventé : on le garde pour que le vigile le refuse
            cits.append(Citation(document_id="INCONNU", element_id=cid, page=0))

    ans = Answer(status=out.status, answer=out.answer, citations=cits,
                 uncertainties=out.uncertainties,
                 clarification_question=out.clarification_question)
    return {"answer": ans.model_dump()}



def node_validate(state):
    _, verdict = check_citations(Answer(**state["answer"]), state["evidence"])
    return {"verdict": verdict}


# ---------- les flèches (décisions) ----------
def route_stop(state):
    return "stop" if "answer" in state else "next"


def route_search(state):
    ev = state.get("evidence", [])
    if ev and ev[0]["rerank_score"] >= MIN_SCORE:
        return "answer"
    if state.get("searches", 0) < MAX_SEARCHES:
        return "search"
    return "clarify"


# ---------- assemblage ----------
g = StateGraph(State)
g.add_node("node_access", node_access)
g.add_node("node_analyze", node_analyze)
g.add_node("node_search", node_search)
g.add_node("node_clarify", node_clarify)
g.add_node("node_answer", node_answer)
g.add_node("node_validate", node_validate)

g.add_edge(START, "node_access")
g.add_conditional_edges("node_access", route_stop,
                        {"stop": END, "next": "node_analyze"})
g.add_conditional_edges("node_analyze", route_stop,
                        {"stop": END, "next": "node_search"})
g.add_conditional_edges("node_search", route_search,
                        {"answer": "node_answer", "search": "node_search",
                         "clarify": "node_clarify"})
g.add_edge("node_answer", "node_validate")
g.add_edge("node_validate", END)
g.add_edge("node_clarify", END)

graph = g.compile()