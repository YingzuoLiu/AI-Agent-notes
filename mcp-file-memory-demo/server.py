from pathlib import Path
from mcp.server.fastmcp import FastMCP
from store import FileMemoryStore

ROOT = Path(__file__).resolve().parent
mcp = FastMCP("file-memory-demo")
store = FileMemoryStore(ROOT / "data" / "docs", ROOT / "data" / "acl.json")

@mcp.tool()
def list_allowed_files(user_id: str):
    """List file-memory documents visible to a user."""
    return {"ok": True, "data": store.list_allowed_files(user_id)}

@mcp.tool()
def search_file_memory(query: str, user_id: str):
    """Search visible file-memory documents with a tiny keyword matcher."""
    results = []
    allowed = store.list_allowed_files(user_id)
    for item in allowed:
        text = Path(item["path"]).read_text(encoding="utf-8")
        if query.lower() in text.lower():
            results.append({
                "doc_id": item["doc_id"],
                "project_id": item["project_id"],
                "snippet": text[:240].replace("\n", " "),
            })
    return {"ok": True, "data": results}

if __name__ == "__main__":
    mcp.run()
