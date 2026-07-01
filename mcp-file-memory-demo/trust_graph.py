def build_trust_graph(documents):
    """Build a tiny provenance graph for demo purposes."""
    nodes = []
    edges = []

    for doc in documents:
        doc_node = f"doc:{doc['doc_id']}"
        project_node = f"project:{doc['project_id']}"
        source_node = f"source:{doc.get('source_type', 'file')}"

        nodes.append({"id": doc_node, "type": "document", "title": doc.get("title", doc["doc_id"])})
        nodes.append({"id": project_node, "type": "project"})
        nodes.append({"id": source_node, "type": "source"})

        edges.append({"from": doc_node, "to": project_node, "type": "belongs_to"})
        edges.append({"from": doc_node, "to": source_node, "type": "derived_from"})

    unique_nodes = {node["id"]: node for node in nodes}
    return {"nodes": list(unique_nodes.values()), "edges": edges}
