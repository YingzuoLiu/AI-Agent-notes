---
doc_id: alpha_runbook
title: Project Alpha Agent Runbook
project_id: alpha
source_type: internal_wiki
version: v3
trust_level: high
---
# Project Alpha Agent Runbook

Project Alpha uses an agent runtime with planner, skill registry, MCP gateway, and sandboxed execution.

The runtime should call tools through MCP instead of directly calling internal APIs. Each tool call should carry user and project context so the service can filter before returning data.

For retrieval, start with lexical and semantic recall, merge candidates, apply access filtering, then return source-aware context to the agent.
