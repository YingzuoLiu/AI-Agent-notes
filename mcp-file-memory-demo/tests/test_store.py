from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from store import FileMemoryStore


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.store = FileMemoryStore(ROOT / "data" / "docs", ROOT / "data" / "acl.json")

    def test_bob_only_sees_public(self):
        files = self.store.list_allowed_files("bob")
        projects = {item["project_id"] for item in files}
        self.assertEqual(projects, {"public"})

    def test_alice_sees_alpha(self):
        files = self.store.list_allowed_files("alice")
        projects = {item["project_id"] for item in files}
        self.assertIn("alpha", projects)


if __name__ == "__main__":
    unittest.main()
