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

    def test_bob_cannot_read_alpha(self):
        result = self.store.read_file_memory("bob", "alpha_runbook")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "access_denied")

    def test_alice_can_read_alpha(self):
        result = self.store.read_file_memory("alice", "alpha_runbook")
        self.assertTrue(result["ok"])
        self.assertIn("agent runtime", result["data"]["content"].lower())

    def test_search_uses_allowed_docs_only(self):
        bob_results = self.store.search_file_memory("agent runtime", "bob")
        self.assertEqual(bob_results, [])
        alice_results = self.store.search_file_memory("agent runtime", "alice")
        self.assertGreaterEqual(len(alice_results), 1)


if __name__ == "__main__":
    unittest.main()
