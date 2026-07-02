from pathlib import Path

from mcp.server.fastmcp import FastMCP

from audit import recent_events
from store import FileMemoryStore
from trust_graph import build_trust_graph

ROOT = Path(__file__).resolve().parent
mcp = FastMCP("file-memory-demo")
store = FileMemoryStore(ROOT / "data" / "docs", ROOT / "data" / "acl.json")


@mcp.tool()
def list_allowed_files(user_id: str, project_id: str | None = None):
    """List file-memory documents visible to a user."""
    return {"ok": True, "data": store.list_allowed_files(user_id, project_id)}


@mcp.tool()
def search_file_memory(query: str, user_id: str, project_id: str | None = None, top_k: int = 5):
    """Search visible file-memory documents with lexical + semantic mock + RRF."""
    return {"ok": True, "data": store.search_file_memory(query, user_id, project_id, top_k)}


@mcp.tool()
def read_file_memory(doc_id: str, user_id: str):
    """Read one file-memory document after access checking."""
    return store.read_file_memory(user_id, doc_id)


@mcp.tool()
def get_trust_graph(user_id: str, project_id: str | None = None):
    """Return a tiny provenance graph for documents visible to a user."""
    docs = store.list_allowed_files(user_id, project_id)
    return {"ok": True, "data": build_trust_graph(docs)}


@mcp.tool()
def get_audit_events(limit: int = 20):
    """Return recent demo audit events."""
    return {"ok": True, "data": recent_events(limit)}


if __name__ == "__main__":
    mcp.run()
