import asyncio
import json
import os
import sys

os.environ.setdefault("APPROVAL_SECRET", "secret-de-test-seulement")

from mcp import Client

import maintenance
from mcp_server import mcp

RESULTS = []


def check(name, cond):
    RESULTS.append(bool(cond))
    print(f"  {'OK    ' if cond else 'ECHEC '} {name}")


def payload(result):
    sc = getattr(result, "structured_content", None)
    if sc:
        return sc["result"] if set(sc) == {"result"} else sc
    return json.loads(result.content[0].text)


async def call(c, name, args):
    return payload(await c.call_tool(name, args))


async def main():
    maintenance.reset()
    eq = "SAN-LAB-2000"
    title, desc = "Alerte AL-01 sur fabric A", "Chemin FC degrade sur HBA1"

    async with Client(mcp, raise_exceptions=True) as c:
        tools = {t.name for t in (await c.list_tools()).tools}
        check("les 3 outils sont exposes", tools == {
            "get_equipment_history", "get_open_tickets", "create_ticket"})

        h = await call(c, "get_equipment_history", {"equipment_id": eq})
        check("historique : 4 interventions", len(h["history"]) == 4)

        h2 = await call(c, "get_equipment_history", {"equipment_id": "INCONNU"})
        check("equipement inconnu : erreur propre", "error" in h2)

        o = await call(c, "get_open_tickets", {"equipment_id": eq})
        check("4 tickets ouverts au depart", len(o["tickets"]) == 4)

        r = await call(c, "create_ticket", {"equipment_id": eq, "title": title,
                       "description": desc, "approval_token": ""})
        check("creation SANS approbation : refusee", r["status"] == "refused")

        r = await call(c, "create_ticket", {"equipment_id": eq, "title": title,
                       "description": desc, "approval_token": "tech1:faux"})
        check("creation avec faux jeton : refusee", r["status"] == "refused")

        token = maintenance.issue_approval("tech1", eq, title, desc)
        r = await call(c, "create_ticket", {"equipment_id": eq, "title": title,
                       "description": desc, "approval_token": token})
        check("creation avec approbation : creee", r["status"] == "created")

        r2 = await call(c, "create_ticket", {"equipment_id": eq, "title": title,
                        "description": desc, "approval_token": token})
        check("meme demande 2 fois : pas de doublon",
              r2["status"] == "already_created"
              and r2["ticket"]["ticket"] == r["ticket"]["ticket"])

        r = await call(c, "create_ticket", {"equipment_id": eq,
                       "title": title + " (modifie)", "description": desc,
                       "approval_token": token})
        check("contenu modifie : nouvelle approbation exigee",
              r["status"] == "refused")

        r = await call(c, "create_ticket", {"equipment_id": eq, "title": title,
                       "description": desc,
                       "approval_token": token.replace("tech1", "tech2", 1)})
        check("jeton d'un autre utilisateur : refuse", r["status"] == "refused")

        o = await call(c, "get_open_tickets", {"equipment_id": eq})
        check("5 tickets ouverts a la fin (1 seul cree)", len(o["tickets"]) == 5)

    ok = sum(RESULTS)
    print(f"\n{ok}/{len(RESULTS)} tests reussis")
    sys.exit(0 if ok == len(RESULTS) else 1)


asyncio.run(main())
