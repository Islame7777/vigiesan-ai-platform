import os
import sys

os.environ.setdefault("APPROVAL_SECRET", "secret-de-test-seulement")

import agent
import maintenance
from llm import LLMOut

if "--llm" not in sys.argv:
    def fake(question, evidence):
        return LLMOut(status="answered", answer="[reponse simulee]",
                      cited_ids=[e["element_id"] for e in evidence])
    agent.ask_llm = fake

RESULTS = []


def check(name, cond):
    RESULTS.append(bool(cond))
    print(f"  {'OK    ' if cond else 'ECHEC '} {name}")


def run(q, **extra):
    return agent.graph.invoke({"question": q, "user_groups": ["technicien"], **extra})


maintenance.reset()

print("\n1. Question du manuel seul")
out = run("Que signifie AL-01 ?")
check("0 appel MCP", out.get("mcp_calls", 0) == 0)
check("reponse donnee", out["answer"]["status"] == "answered")
check("citations validees", out["verdict"].startswith("OK"))

print("\n2. Question sur un antecedent")
out = run("Le disque B03 a-t-il un antecedent ?")
ids = [e["element_id"] for e in out.get("mcp_evidence", [])]
print("   statut :", out["answer"]["status"], "| verdict :", out.get("verdict"))
check("2 appels MCP", out.get("mcp_calls", 0) == 2)
check("INC-117 dans les preuves", "INC-117" in ids)
cited = [c["element_id"] for c in out["answer"].get("citations", [])]
check("INC-117 cite et valide", "INC-117" in cited and out["verdict"].startswith("OK"))

print("\n3. Demande de ticket")
before = len(maintenance.open_tickets("SAN-LAB-2000")["tickets"])
out = run("Prepare un ticket pour l'alarme AL-01 sur la fabric A")
draft = out.get("ticket_draft")
check("brouillon prepare", draft is not None)
check("titre avec le code AL-01", draft is not None and "AL-01" in draft["title"])
check("rien cree par l'agent",
      len(maintenance.open_tickets("SAN-LAB-2000")["tickets"]) == before)
check("budget MCP : 2 appels + 1 creation <= 3",
      out.get("mcp_calls", 0) + 1 <= agent.MAX_MCP_CALLS)
if draft:
    r = agent.create_ticket_after_approval(draft, "tech1")
    check("creation apres approbation", r["status"] == "created")
    r2 = agent.create_ticket_after_approval(draft, "tech1")
    check("pas de doublon", r2["status"] == "already_created")
    check("1 seul ticket de plus",
          len(maintenance.open_tickets("SAN-LAB-2000")["tickets"]) == before + 1)

print("\n4. Question sur la procedure de creation (pas une demande)")
out = run("Qui doit approuver la creation du ticket ?")
check("0 appel MCP", out.get("mcp_calls", 0) == 0)
check("pas de brouillon", "ticket_draft" not in out)

print("\n5. Budget MCP deja consomme")
out = run("Le disque B03 a-t-il un antecedent ?", mcp_calls=2)
check("aucun appel de plus", out.get("mcp_calls", 0) == 2)
check("pas de preuve MCP", not out.get("mcp_evidence"))

ok = sum(RESULTS)
print(f"\n{ok}/{len(RESULTS)} tests reussis")
sys.exit(0 if ok == len(RESULTS) else 1)
