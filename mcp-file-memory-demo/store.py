from pathlib import Path
import json

class FileMemoryStore:
    def __init__(self, docs_dir="data/docs", acl_path="data/acl.json"):
        self.docs_dir = Path(docs_dir)
        self.acl = json.loads(Path(acl_path).read_text(encoding="utf-8"))

    def can_read(self, user_id, project_id):
        return project_id in self.acl.get(user_id, [])

    def list_allowed_files(self, user_id):
        files = []
        for path in self.docs_dir.glob("*.md"):
            text = path.read_text(encoding="utf-8")
            project_id = "public" if "project_id: public" in text else "alpha"
            if self.can_read(user_id, project_id):
                files.append({"doc_id": path.stem, "project_id": project_id, "path": str(path)})
        return files
