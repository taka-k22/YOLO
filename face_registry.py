"""Persistent face embeddings and a small DeepFace adapter.

The registry deliberately stores embeddings only. Camera frames and face crops stay
in memory and are never written to disk by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import threading
import uuid
from typing import Iterable, Optional


REGISTRY_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalise_embedding(values: Iterable[float]) -> list[float]:
    embedding = [float(value) for value in values]
    if not embedding or not all(math.isfinite(value) for value in embedding):
        raise ValueError("embedding must contain finite numbers")

    norm = math.sqrt(sum(value * value for value in embedding))
    if norm <= 0:
        raise ValueError("embedding norm must be positive")
    return [value / norm for value in embedding]


@dataclass(frozen=True)
class FaceMatch:
    person_id: str
    name: str
    distance: float
    threshold: float


class FaceRegistry:
    """Thread-safe JSON registry for named face embeddings."""

    def __init__(
        self,
        path: str | Path,
        *,
        model_name: str,
        threshold: float,
        max_embeddings_per_person: int = 5,
    ) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        if max_embeddings_per_person < 1:
            raise ValueError("max_embeddings_per_person must be at least 1")

        self.path = Path(path)
        self.model_name = model_name
        self.threshold = float(threshold)
        self.max_embeddings_per_person = max_embeddings_per_person
        self._lock = threading.RLock()
        self._people: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return

        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("version") != REGISTRY_VERSION:
            raise ValueError(f"unsupported face registry version: {payload.get('version')}")
        if payload.get("model") != self.model_name:
            raise ValueError(
                "face registry model does not match configuration: "
                f"{payload.get('model')} != {self.model_name}"
            )

        people = payload.get("people")
        if not isinstance(people, list):
            raise ValueError("face registry people must be a list")

        loaded: dict[str, dict] = {}
        for person in people:
            person_id = str(person["person_id"])
            name = str(person["name"]).strip()
            embeddings = [
                _normalise_embedding(embedding)
                for embedding in person.get("embeddings", [])
            ]
            if not person_id or not name or not embeddings:
                continue
            loaded[person_id] = {
                "person_id": person_id,
                "name": name,
                "created_at": person.get("created_at") or _utc_now(),
                "updated_at": person.get("updated_at") or _utc_now(),
                "embeddings": embeddings[-self.max_embeddings_per_person :],
            }
        self._people = loaded

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": REGISTRY_VERSION,
            "model": self.model_name,
            "distance_metric": "cosine",
            "people": list(self._people.values()),
        }
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(self.path)

    def list_people(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "person_id": person["person_id"],
                    "name": person["name"],
                    "created_at": person["created_at"],
                    "updated_at": person["updated_at"],
                    "embedding_count": len(person["embeddings"]),
                }
                for person in sorted(
                    self._people.values(), key=lambda item: item["name"].casefold()
                )
            ]

    def match(self, embedding: Iterable[float]) -> Optional[FaceMatch]:
        candidate = _normalise_embedding(embedding)
        best: Optional[FaceMatch] = None

        with self._lock:
            for person in self._people.values():
                for reference in person["embeddings"]:
                    if len(candidate) != len(reference):
                        continue
                    similarity = sum(
                        left * right for left, right in zip(candidate, reference)
                    )
                    distance = 1.0 - max(-1.0, min(1.0, similarity))
                    if best is None or distance < best.distance:
                        best = FaceMatch(
                            person_id=person["person_id"],
                            name=person["name"],
                            distance=distance,
                            threshold=self.threshold,
                        )

        if best is None or best.distance > self.threshold:
            return None
        return best

    def enroll(self, name: str, embedding: Iterable[float]) -> dict:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("name must not be empty")

        normalised = _normalise_embedding(embedding)
        now = _utc_now()

        with self._lock:
            person = next(
                (
                    item
                    for item in self._people.values()
                    if item["name"].casefold() == clean_name.casefold()
                ),
                None,
            )
            if person is None:
                person_id = uuid.uuid4().hex
                person = {
                    "person_id": person_id,
                    "name": clean_name,
                    "created_at": now,
                    "updated_at": now,
                    "embeddings": [],
                }
                self._people[person_id] = person

            person["name"] = clean_name
            person["updated_at"] = now
            person["embeddings"].append(normalised)
            person["embeddings"] = person["embeddings"][-self.max_embeddings_per_person :]
            self._save_locked()

            return {
                "person_id": person["person_id"],
                "name": person["name"],
                "embedding_count": len(person["embeddings"]),
            }


class DeepFaceEncoder:
    """Lazy DeepFace wrapper so tracking can still run when DeepFace is unavailable."""

    def __init__(self, *, model_name: str, detector_backend: str) -> None:
        from deepface import DeepFace  # type: ignore

        self._deepface = DeepFace
        self.model_name = model_name
        self.detector_backend = detector_backend
        self._deepface.build_model(model_name=model_name)

    def encode(self, image) -> list[float]:
        representations = self._deepface.represent(
            img_path=image,
            model_name=self.model_name,
            detector_backend=self.detector_backend,
            enforce_detection=True,
            align=True,
            max_faces=None,
        )
        if not representations:
            raise ValueError("no face detected")

        largest = max(
            representations,
            key=lambda item: (
                item.get("facial_area", {}).get("w", 0)
                * item.get("facial_area", {}).get("h", 0)
            ),
        )
        return _normalise_embedding(largest["embedding"])
