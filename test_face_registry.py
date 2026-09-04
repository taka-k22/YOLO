import json
import tempfile
import unittest
from pathlib import Path

from face_registry import FaceRegistry


class FaceRegistryTests(unittest.TestCase):
    def make_registry(self, directory: str) -> FaceRegistry:
        return FaceRegistry(
            Path(directory) / "faces.json",
            model_name="test-model",
            threshold=0.2,
            max_embeddings_per_person=2,
        )

    def test_enroll_match_and_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = self.make_registry(directory)
            enrolled = registry.enroll("Alice", [1.0, 0.0, 0.0])

            match = registry.match([0.99, 0.01, 0.0])
            self.assertIsNotNone(match)
            self.assertEqual(match.person_id, enrolled["person_id"])
            self.assertEqual(match.name, "Alice")

            reloaded = self.make_registry(directory)
            self.assertEqual(reloaded.list_people()[0]["name"], "Alice")

    def test_unknown_embedding_does_not_match(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = self.make_registry(directory)
            registry.enroll("Alice", [1.0, 0.0])
            self.assertIsNone(registry.match([0.0, 1.0]))

    def test_same_name_adds_limited_exemplars(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = self.make_registry(directory)
            first = registry.enroll("Alice", [1.0, 0.0, 0.0])
            second = registry.enroll("alice", [0.99, 0.01, 0.0])
            third = registry.enroll("Alice", [0.98, 0.02, 0.0])

            self.assertEqual(first["person_id"], second["person_id"])
            self.assertEqual(second["person_id"], third["person_id"])
            self.assertEqual(third["embedding_count"], 2)

            payload = json.loads(
                (Path(directory) / "faces.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(payload["people"][0]["embeddings"]), 2)


if __name__ == "__main__":
    unittest.main()
