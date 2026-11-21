from typing import Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    document_id: str
    element_id: str
    page: int


class Answer(BaseModel):
    status: Literal["answered", "needs_clarification"]
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    clarification_question: str | None = None


def check_citations(answer: Answer, evidence: list[dict]):
    """Chaque citation doit exister dans les preuves réellement récupérées."""
    allowed = {(e["document_id"], e["element_id"], e["page"]) for e in evidence}
    results = [(c, (c.document_id, c.element_id, c.page) in allowed)
               for c in answer.citations]
    if answer.status == "answered" and not answer.citations:
        return results, "Réponse sans aucune citation : refusée"
    if any(not ok for _, ok in results):
        return results, "Citation inventée ou incorrecte : refusée"
    return results, "OK (structure seulement)"