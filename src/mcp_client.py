"""Client MCP : appelle les outils du serveur de maintenance."""
import asyncio
import json

from mcp import Client

from mcp_server import mcp as _server


def _payload(result):
    if getattr(result, "is_error", False):
        text = result.content[0].text if result.content else "erreur outil"
        raise RuntimeError(text)
    sc = getattr(result, "structured_content", None)
    if sc:
        return sc["result"] if set(sc) == {"result"} else sc
    return json.loads(result.content[0].text)


async def _call(name, args):
    async with Client(_server) as c:
        return _payload(await c.call_tool(name, args))


def mcp_call(name, args):
    """Appelle un outil MCP (serveur en memoire, meme processus)."""
    return asyncio.run(_call(name, args))
