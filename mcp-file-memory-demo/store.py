from pathlib import Path
import json

from audit import record_event
from retrieval import lexical_rank, semantic_mock_rank, rrf_fuse


class FileMemoryStore:
    def __init__(self, docs_dir="data/docs", acl_path="data/acl.json"):
        self.docs_dir = Path(docs_dir)
        self.acl = json.loads(Path(acl_path).read_text(encoding="utf-8"))
        self.docs = self.load_docs()

    def parse_doc(self, path):
        text = path.read_text(encoding="utf-8")
        meta = {}
        body = text
        if text.startswith("---"):
            _blank, front, body = text.split("---", 2)
            for line in front.strip().splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    meta[key.strip()] = value.strip()
        return {
            "doc_id": meta.get("doc_id", path.stem),
            "title": meta.get("title", path.stem.replace("_", " ").title()),
            "project_id": meta.get("project_id", "public"),
            "source_type": meta.get("source_type", "file"),
            "version": meta.get("version", "v1"),
            "trust_level": meta.get("trust_level", "medium"),
            "path": str(path),
            "content": body.strip(),
        }

    def load_docs(self):
        docs = {}
        for path in self.docs_dir.glob("*.md"):
            doc = self.parse_doc(path)
            docs[doc["doc_id"]] = doc
        return docs

    def can_read(self, user_id, project_id):
        return project_id in self.acl.get(user_id, [])

    def public_view(self, doc):
        return {
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "project_id": doc["project_id"],
            "source_type": doc["source_type"],
            "version": doc["version"],
            "trust_level": doc["trust_level"],
            "path": doc["path"],
        }

    def list_allowed_files(self, user_id, project_id=None):
        files = []
        for doc in self.docs.values():
            if project_id and doc["project_id"] != project_id:
                continue
            allowed = self.can_read(user_id, doc["project_id"])
            record_event(user_id, "list", doc["doc_id"], allowed)
            if allowed:
                files.append(self.public_view(doc))
        return files

    def read_file_memory(self, user_id, doc_id):
        doc = self.docs.get(doc_id)
        if doc is None:
            return {"ok": False, "error": "not_found", "message": "document not found"}
        allowed = self.can_read(user_id, doc["project_id"])
        record_event(user_id, "read", doc_id, allowed)
        if not allowed:
            return {"ok": False, "error": "access_denied", "message": "user cannot read this document"}
        return {"ok": True, "data": {"content": doc["content"], "metadata": self.public_view(doc)}}

    def search_file_memory(self, query, user_id, project_id=None, top_k=5):
        visible_docs = []
        for doc in self.docs.values():
            if project_id and doc["project_id"] != project_id:
                continue
            allowed = self.can_read(user_id, doc["project_id"])
            record_event(user_id, "search_candidate", doc["doc_id"], allowed)
            if allowed:
                visible_docs.append(doc)

        lexical = lexical_rank(query, visible_docs)
        semantic = semantic_mock_rank(query, visible_docs)
        fused = rrf_fuse([lexical, semantic])[:top_k]

        results = []
        for doc_id in fused:
            doc = self.docs[doc_id]
            results.append({
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "project_id": doc["project_id"],
                "snippet": doc["content"][:240].replace("\n", " "),
                "metadata": self.public_view(doc),
            })
        return results
