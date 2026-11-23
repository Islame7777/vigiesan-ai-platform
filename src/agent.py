import re
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from search import search_text, rerank
from citations import Answer, Citation, check_citations
from llm import ask_llm
from mcp_client import mcp_call
from maintenance import issue_approval

MAX_SEARCHES = 2        # limite de boucle : jamais plus de 2 recherches
MAX_MCP_CALLS = 3       # budget du sujet : 3 appels MCP par demande
MIN_SCORE = 0.05        # seuil de depart, a mesurer a l'etape 5
DEFAULT_EQUIPMENT = "SAN-LAB-2000"
ALL_GROUPS = ["technicien", "responsable"]

ALARM_CODE = re.compile(r"\bAL[- ]?\d{2}\b", re.IGNORECASE)
VOYANT = re.compile(r"\bvoyants?\b|\bLED\b", re.IGNORECASE)
NEEDS_HISTORY = re.compile(
    r"ant[ée]c[ée]dent|historique|d[ée]j[àa]\b|\bINC-?\d+|\bB0\d\b|"
    r"disque|fabric|contr[ôo]leur|volume|VOL-DB|capacit[ée]|r[ée]current",
    re.IGNORECASE)
WANTS_TICKET = re.compile(
    r"\b(pr[ée]pare\w*|cr[ée]e[rz]?|ouvre|ouvrir|g[ée]n[èe]re\w*)\s+(un\s+)?ticket",
    re.IGNORECASE)


class State(TypedDict, total=False):
    question: str
    user_groups: list
    equipment_id: str
    evidence: list
    mcp_evidence: list
    mcp_calls: int
    mcp_error: str
    searches: int
    answer: dict
    verdict: str
    ticket_draft: dict


def all_evidence(state):
    return list(state.get("evidence", [])) + list(state.get("mcp_evidence", []))


# ---------- les cases du graphe ----------
def node_access(state):
    if not state.get("user_groups"):
        return {"answer": {"status": "refused",
                           "answer": "Acces refuse : aucun groupe d'acces.",
                           "citations": []},
                "verdict": "refuse"}
    return {}


def node_analyze(state):
    q = state["question"]
    if VOYANT.search(q) and not ALARM_CODE.search(q):
        return {"answer": {
            "status": "needs_clarification",
            "answer": "Un voyant seul ne permet pas d'identifier la cause.",
            "clarification_question": "Quel est le code d'alarme affiche, "
                                      "le modele de la baie et le contexte du panneau ?",
            "citations": []},
            "verdict": "clarification demandee"}
    return {}


def node_history(state):
    """Consulte l'historique via MCP, seulement si la question le justifie."""
    q = state["question"]
    if not (NEEDS_HISTORY.search(q) or WANTS_TICKET.search(q)):
        return {}
    used = state.get("mcp_calls", 0)
    if used + 2 > MAX_MCP_CALLS:
        return {"mcp_error": "budget d'appels MCP epuise"}
    eq = state.get("equipment_id", DEFAULT_EQUIPMENT)
    try:
        hist = mcp_call("get_equipment_history", {"equipment_id": eq})
        used += 1
        opened = mcp_call("get_open_tickets", {"equipment_id": eq})
        used += 1
    except Exception as e:
        return {"mcp_calls": used, "mcp_error": type(e).__name__}

    open_ids = {t["ticket"] for t in opened.get("tickets", [])}
    rows = hist.get("history", [])
    ql = q.lower()

    def relevant(r):
        toks = [t for t in re.split(r"\W+", r["asset"].lower()) if len(t) >= 3]
        return any(t in ql for t in toks) or r["ticket"].lower() in ql

    picked = [r for r in rows if relevant(r)] or rows
    mcp_ev = []
    for r in picked:
        statut = "ouvert" if r["ticket"] in open_ids else r.get("status", "?")
        mcp_ev.append({
            "document_id": "MCP-HISTORIQUE",
            "element_id": r["ticket"],
            "page": 0,
            "type": "historique_mcp",
            "access_groups": ALL_GROUPS,
            "text": (f"{r['ticket']} | {r['date']} | {r['asset']} : "
                     f"{r['observation']} | statut : {statut}"),
        })
    return {"mcp_calls": used, "mcp_evidence": mcp_ev}


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
        "answer": "Je n'ai pas trouve de source assez pertinente "
                  "dans les documents autorises.",
        "clarification_question": "Pouvez-vous reformuler ou preciser "
                                  "l'equipement et le code d'alarme ?",
        "citations": []},
        "verdict": "clarification : preuves insuffisantes"}


def node_answer(state):
    ev = all_evidence(state)
    extra = []
    if state.get("mcp_error"):
        extra.append("Historique des interventions indisponible : "
                     + state["mcp_error"])
    try:
        out = ask_llm(state["question"], ev)
    except Exception as e:
        return {"answer": {
            "status": "needs_clarification",
            "answer": "Le modele de langage est indisponible.",
            "citations": [],
            "uncertainties": [type(e).__name__]},
            "verdict": "erreur modele"}

    by_id = {e["element_id"]: e for e in ev}
    cits = []
    for cid in out.cited_ids:
        e = by_id.get(cid)
        if e:
            cits.append(Citation(document_id=e["document_id"],
                                 element_id=cid, page=e["page"]))
        else:   # identifiant invente : on le garde pour que le vigile le refuse
            cits.append(Citation(document_id="INCONNU", element_id=cid, page=0))

    ans = Answer(status=out.status, answer=out.answer, citations=cits,
                 uncertainties=list(out.uncertainties) + extra,
                 clarification_question=out.clarification_question)
    return {"answer": ans.model_dump()}


def node_validate(state):
    _, verdict = check_citations(Answer(**state["answer"]), all_evidence(state))
    return {"verdict": verdict}


def node_draft(state):
    """Prepare un BROUILLON de ticket. Rien n'est cree ici."""
    q = state["question"]
    ev = all_evidence(state)
    if not WANTS_TICKET.search(q) or not ev:
        return {}
    eq = state.get("equipment_id", DEFAULT_EQUIPMENT)
    m = ALARM_CODE.search(q)
    if m:
        code = re.sub(r"^AL[- ]?", "AL-", m.group(0).upper())
        title = f"Alerte {code} sur {eq}"
    else:
        title = f"Demande de maintenance sur {eq}"
    ids = ", ".join(e["element_id"] for e in ev[:6])
    description = (f"Question du technicien : {q.strip()} | Preuves : {ids} | "
                   f"Brouillon : approbation humaine requise.")
    return {"ticket_draft": {
        "equipment_id": eq,
        "title": title.strip()[:120],
        "description": description.strip(),
        "status": "Brouillon - approbation requise"}}


def create_ticket_after_approval(draft, user):
    """A appeler UNIQUEMENT quand l'humain a clique sur Approuver.
    Signe l'approbation (liee a l'utilisateur et au contenu), puis appelle
    l'outil MCP create_ticket."""
    token = issue_approval(user, draft["equipment_id"],
                           draft["title"], draft["description"])
    return mcp_call("create_ticket", {
        "equipment_id": draft["equipment_id"],
        "title": draft["title"],
        "description": draft["description"],
        "approval_token": token})


# ---------- les fleches (decisions) ----------
def route_stop(state):
    return "stop" if "answer" in state else "next"


def route_search(state):
    ev = state.get("evidence", [])
    if ev and ev[0]["rerank_score"] >= MIN_SCORE:
        return "answer"
    if state.get("mcp_evidence"):
        return "answer"
    if state.get("searches", 0) < MAX_SEARCHES:
        return "search"
    return "clarify"


# ---------- assemblage ----------
g = StateGraph(State)
g.add_node("node_access", node_access)
g.add_node("node_analyze", node_analyze)
g.add_node("node_history", node_history)
g.add_node("node_search", node_search)
g.add_node("node_clarify", node_clarify)
g.add_node("node_answer", node_answer)
g.add_node("node_validate", node_validate)
g.add_node("node_draft", node_draft)

g.add_edge(START, "node_access")
g.add_conditional_edges("node_access", route_stop,
                        {"stop": END, "next": "node_analyze"})
g.add_conditional_edges("node_analyze", route_stop,
                        {"stop": END, "next": "node_history"})
g.add_edge("node_history", "node_search")
g.add_conditional_edges("node_search", route_search,
                        {"answer": "node_answer", "search": "node_search",
                         "clarify": "node_clarify"})
g.add_edge("node_answer", "node_validate")
g.add_edge("node_validate", "node_draft")
g.add_edge("node_draft", END)
g.add_edge("node_clarify", END)

graph = g.compile()
