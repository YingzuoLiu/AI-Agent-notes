# File Memory MCP Demo

A small Model Context Protocol demo for agent platform practice.

This demo shows how to expose local file memory through MCP tools with structured inputs, structured outputs, permission checks, source metadata, and simple retrieval.

## Components

- MCP server with three tools
- Local markdown document store
- User/project permission table
- Structured error handling
- Unit tests for retrieval and access control

## Tools

- `list_allowed_files(user_id, project_id=None)`
- `search_file_memory(query, user_id, project_id=None, top_k=5)`
- `read_file_memory(doc_id, user_id)`

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m file_memory_mcp_demo.server
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
python -m file_memory_mcp_demo.server
```

## Test

```bash
python -m unittest discover -s tests
```

## Why this matters

In an enterprise agent platform, MCP integration is not only a function wrapper. A useful tool boundary also needs clear schemas, permission checks, provenance metadata, safe error messages, and traceable outputs.

A next step would be replacing the toy lexical search with hybrid retrieval: BM25, vector retrieval, RRF, reranking, and trust-graph expansion.
