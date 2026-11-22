"""Serveur MCP VigieSAN : historique, tickets ouverts, creation de ticket."""
from typing import Any

try:
    from mcp.server import MCPServer
except ImportError:  # selon la version de mcp 2.x
    from mcp.server.mcpserver import MCPServer

import maintenance

mcp = MCPServer(
    "VigieSAN-maintenance",
    instructions=("Outils de maintenance SAN (donnees fictives). "
                  "create_ticket exige un jeton d'approbation humaine."),
)


@mcp.tool()
def get_equipment_history(equipment_id: str) -> dict[str, Any]:
    """Historique des interventions d'un equipement (lecture seule)."""
    return maintenance.history(equipment_id)


@mcp.tool()
def get_open_tickets(equipment_id: str) -> dict[str, Any]:
    """Tickets ouverts d'un equipement (lecture seule)."""
    return maintenance.open_tickets(equipment_id)


@mcp.tool()
def create_ticket(equipment_id: str, title: str, description: str,
                  approval_token: str) -> dict[str, Any]:
    """Cree un ticket. Refuse sans jeton d'approbation humaine valide,
    lie a l'utilisateur et au contenu exact."""
    return maintenance.create_ticket(equipment_id, title, description,
                                     approval_token)


if __name__ == "__main__":
    mcp.run()
