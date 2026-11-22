"""Logique de maintenance (donnees FICTIVES du manuel, section 6)."""
import copy
import hashlib
import hmac
import os

import config  # noqa: F401  (charge le fichier .env)

EQUIPMENTS = {"SAN-LAB-2000"}
MAX_TITLE, MAX_DESC = 120, 2000

# Historique repris de la section 6 du manuel. Le manuel ne donne pas de
# statut ouvert/ferme : par simplification, tous sont "ouvert" ici.
_SEED = [
    {"ticket": "INC-104", "date": "2026-09-18", "asset": "Fabric A",
     "observation": "Perte temporaire de chemin HBA1 ; surveillance"},
    {"ticket": "INC-117", "date": "2026-10-02", "asset": "Disque B03",
     "observation": "Pre-alerte ; examen du groupe de protection"},
    {"ticket": "INC-125", "date": "2026-10-20", "asset": "Volume VOL-DB",
     "observation": "Capacite a 88 % ; analyse de tendance"},
    {"ticket": "INC-130", "date": "2026-10-25", "asset": "Controleur B",
     "observation": "Evenement non reproductible ; suivi"},
]

_tickets = []
_by_key = {}
_next = 131


def reset():
    """Remet l'etat de depart (utile pour les tests)."""
    global _tickets, _by_key, _next
    _tickets = [{**copy.deepcopy(t), "equipment_id": "SAN-LAB-2000",
                 "status": "ouvert"} for t in _SEED]
    _by_key = {}
    _next = 131


reset()


def _norm(equipment_id):
    return str(equipment_id).strip().upper()


def _secret():
    s = os.environ.get("APPROVAL_SECRET")
    if not s:
        raise RuntimeError("APPROVAL_SECRET manquant (a mettre dans .env)")
    return s.encode()


def _sign(user, equipment_id, title, description):
    msg = "|".join([user, equipment_id, title, description]).encode()
    return hmac.new(_secret(), msg, hashlib.sha256).hexdigest()


def issue_approval(user, equipment_id, title, description):
    """Appelee par l'APPLICATION quand l'humain clique sur Approuver.
    Ce n'est PAS un outil MCP : le modele ne peut pas s'approuver."""
    eq = _norm(equipment_id)
    return f"{user}:{_sign(user, eq, title, description)}"


def history(equipment_id):
    eq = _norm(equipment_id)
    if eq not in EQUIPMENTS:
        return {"equipment_id": eq, "error": "equipement inconnu", "history": []}
    rows = sorted((t for t in _tickets if t["equipment_id"] == eq),
                  key=lambda t: t["date"])
    return {"equipment_id": eq, "history": rows}


def open_tickets(equipment_id):
    eq = _norm(equipment_id)
    if eq not in EQUIPMENTS:
        return {"equipment_id": eq, "error": "equipement inconnu", "tickets": []}
    rows = [t for t in _tickets
            if t["equipment_id"] == eq and t["status"] == "ouvert"]
    return {"equipment_id": eq, "tickets": rows}


def _refuse(reason):
    return {"status": "refused", "reason": reason}


def create_ticket(equipment_id, title, description, approval_token):
    global _next
    eq = _norm(equipment_id)
    title, description = str(title).strip(), str(description).strip()
    if eq not in EQUIPMENTS:
        return _refuse("equipement inconnu")
    if not title or not description:
        return _refuse("titre et description obligatoires")
    if len(title) > MAX_TITLE or len(description) > MAX_DESC:
        return _refuse("titre ou description trop long")
    if not approval_token or ":" not in approval_token:
        return _refuse("approbation humaine absente")

    user, sig = approval_token.rsplit(":", 1)
    expected = _sign(user, eq, title, description)
    if not hmac.compare_digest(sig, expected):
        return _refuse("approbation invalide ou contenu modifie")

    key = hashlib.sha256("|".join([user, eq, title, description]).encode()).hexdigest()
    if key in _by_key:
        return {"status": "already_created", "ticket": _by_key[key]}

    ticket = {"ticket": f"INC-{_next}", "date": "2026-11-22", "asset": "",
              "observation": title, "equipment_id": eq,
              "status": "ouvert", "approved_by": user,
              "description": description}
    _next += 1
    _tickets.append(ticket)
    _by_key[key] = ticket
    return {"status": "created", "ticket": ticket}
