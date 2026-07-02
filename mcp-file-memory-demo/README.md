# File Memory MCP Demo

A small Model Context Protocol demo for agent platform practice.

This project shows how an agent can use file memory through MCP tools without reading every file directly. The MCP server checks user/project access first, then returns structured results with metadata.

## What this demo covers

- MCP tools as a safe boundary between agent runtime and internal files
- Local markdown document store
- User/project ACL
- Read/search/list tools
- Source metadata such as project, source type, version, and trust level
- Tiny retrieval fusion: lexical rank + semantic mock rank + RRF
- Tiny trust graph for provenance-style relationships
- In-memory audit events for tool calls
- Unit tests for access control and retrieval behavior

## Tools exposed by the MCP server

- `list_allowed_files(user_id, project_id=None)`
- `search_file_memory(query, user_id, project_id=None, top_k=5)`
- `read_file_memory(doc_id, user_id)`
- `get_trust_graph(user_id, project_id=None)`
- `get_audit_events(limit=20)`

## Architecture

```text
Agent Runtime
  -> MCP Client
    -> file-memory-demo MCP Server
      -> FileMemoryStore
        -> ACL check
        -> retrieval fusion
        -> local markdown documents
      -> Trust Graph helper
      -> Audit helper
```

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python server.py
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
python server.py
```

## Test

```bash
python -m unittest discover -s tests
```

## Why this matters

In an enterprise agent platform, MCP integration is not only a function wrapper. A useful tool boundary also needs clear schemas, access checks, provenance metadata, safe errors, retrieval control, and traceable outputs.

This demo is intentionally small, but it maps to a real enterprise pattern: before an agent can search or read memory, the server checks which project files the user is allowed to see. Retrieval only runs on visible documents. Search results carry metadata so later components can cite, filter, or build a trust graph.

A next step would be replacing the toy semantic mock with real embeddings, adding BM25, a reranker, persistent audit logs, and a stronger trust graph.
