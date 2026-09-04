from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
import os
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time
from typing import Optional
import uuid

import cv2  # type: ignore
from deep_sort_realtime.deepsort_tracker import DeepSort  # type: ignore
from flask import Flask, Response, jsonify, request  # type: ignore
import requests  # type: ignore
from ultralytics import YOLO  # type: ignore

from face_registry import DeepFaceEncoder, FaceRegistry


BASE_DIRECTORY = Path(__file__).resolve().parent
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


EVENT_SEND_ENABLED = env_bool("VISION_EVENT_SEND_ENABLED", True)
DISPLAY_ENABLED = env_bool("VISION_DISPLAY_ENABLED", True)
FACE_RECOGNITION_ENABLED = env_bool("FACE_RECOGNITION_ENABLED", True)
FACE_EVENT_SEND_DEFAULT_ENABLED = env_bool("FACE_EVENT_SEND_ENABLED", True)

EVENT_URL = os.getenv("VISION_EVENT_URL", "http://localhost:3000/yolo_event")
EVENT_REQUEST_TIMEOUT_SECONDS = float(os.getenv("VISION_EVENT_REQUEST_TIMEOUT_SECONDS", "10.0"))
HTTP_HOST = os.getenv("VISION_HTTP_HOST", "127.0.0.1")
HTTP_PORT = int(os.getenv("VISION_HTTP_PORT", "5000"))
CAMERA_INDEX = int(os.getenv("VISION_CAMERA_INDEX", "0"))
YOLO_MODEL_PATH = os.getenv("VISION_YOLO_MODEL", str(BASE_DIRECTORY / "yolov8n.pt"))

FACE_MODEL_NAME = os.getenv("FACE_MODEL_NAME", "SFace")
FACE_DETECTOR_BACKEND = os.getenv("FACE_DETECTOR_BACKEND", "opencv")
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.55"))
FACE_ANALYSIS_INTERVAL_SECONDS = float(os.getenv("FACE_ANALYSIS_INTERVAL_SECONDS", "1.0"))
FACE_CONFIRMATIONS = int(os.getenv("FACE_CONFIRMATIONS", "2"))
FACE_UNKNOWN_CONFIRMATIONS = int(os.getenv("FACE_UNKNOWN_CONFIRMATIONS", "5"))
FACE_MIN_PERSON_WIDTH = int(os.getenv("FACE_MIN_PERSON_WIDTH", "100"))
FACE_MIN_PERSON_HEIGHT = int(os.getenv("FACE_MIN_PERSON_HEIGHT", "140"))
TRACK_EVENT_STABILITY_SECONDS = float(os.getenv("TRACK_EVENT_STABILITY_SECONDS", "1.5"))
TRACK_IDENTITY_WAIT_SECONDS = float(os.getenv("TRACK_IDENTITY_WAIT_SECONDS", "6.0"))
FACE_REGISTRY_PATH = Path(
    os.getenv(
        "FACE_REGISTRY_PATH",
        str(BASE_DIRECTORY / "data" / "face_registry.json"),
    )
)

app = Flask(__name__)
frame_lock = threading.Lock()
latest_frame = None
latest_annotated_frame = None
event_queue: Queue[dict] = Queue(maxsize=100)
face_event_state_lock = threading.RLock()
face_event_send_enabled = FACE_EVENT_SEND_DEFAULT_ENABLED


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def face_events_enabled() -> bool:
    with face_event_state_lock:
        return EVENT_SEND_ENABLED and face_event_send_enabled


def set_face_events_enabled(enabled: bool) -> dict:
    global face_event_send_enabled

    with face_event_state_lock:
        face_event_send_enabled = enabled

    dropped = 0
    if not enabled:
        while True:
            try:
                event_queue.get_nowait()
            except Empty:
                break
            event_queue.task_done()
            dropped += 1

    state = {
        "enabled": face_event_send_enabled,
        "effective_enabled": face_events_enabled(),
        "master_event_delivery_enabled": EVENT_SEND_ENABLED,
        "queued_events_dropped": dropped,
    }
    print("face event delivery changed:", state)
    return state


def pending_identity(status: str = "pending") -> dict:
    return {
        "status": status,
        "person_id": None,
        "name": None,
        "distance": None,
        "threshold": FACE_MATCH_THRESHOLD,
    }


def event_message(event_type: str, identity: dict) -> str:
    name = identity.get("name")
    messages = {
        "person_appeared": "A person has appeared. Identity recognition is in progress.",
        "person_unknown": "A visible face is not registered.",
        "person_disappeared": "A tracked person has disappeared.",
        "person_enrolled": f"The visible person was registered as {name}.",
        "person_recognized": f"The visible registered person is {name}.",
    }
    return messages[event_type]


def send_person_event(
    event_type: str,
    track_id: str,
    *,
    position: Optional[str] = None,
    identity: Optional[dict] = None,
) -> bool:
    identity_payload = identity or pending_identity(
        "pending" if FACE_RECOGNITION_ENABLED else "unavailable"
    )
    event = {
        "event_id": uuid.uuid4().hex,
        "source": "deepsort",
        "type": event_type,
        "track_id": str(track_id),
        "timestamp": utc_now(),
        "identity": identity_payload,
        "message": event_message(event_type, identity_payload),
    }
    if position is not None:
        event["position"] = position

    payload = {"event": event}
    if not face_events_enabled():
        print(
            "face event skipped: disabled",
            {"type": event_type, "track_id": str(track_id)},
        )
        return False

    try:
        event_queue.put_nowait(payload)
        return True
    except Full:
        print("vision event dropped: delivery queue is full", payload)
        return False


def run_event_sender() -> None:
    retry_delays = (0.0, 0.2, 0.5, 1.0)
    while True:
        payload = event_queue.get()
        if not face_events_enabled():
            print(
                "queued face event dropped: disabled",
                payload.get("event", {}).get("event_id"),
            )
            event_queue.task_done()
            continue

        delivered = False
        last_error = None
        for delay in retry_delays:
            if delay:
                time.sleep(delay)
            if not face_events_enabled():
                break
            try:
                response = requests.post(
                    EVENT_URL,
                    json=payload,
                    timeout=EVENT_REQUEST_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                print("vision event sent:", payload)
                delivered = True
                break
            except requests.RequestException as error:
                last_error = error
        if not delivered and face_events_enabled():
            print("vision event failed after retries:", last_error, payload)
        elif not delivered:
            print(
                "face event delivery stopped: disabled",
                payload.get("event", {}).get("event_id"),
            )
        event_queue.task_done()


threading.Thread(target=run_event_sender, daemon=True, name="vision-events").start()


class FaceRecognitionService:
    """Runs DeepFace on one background thread and binds identities to tracks."""

    def __init__(self, registry: FaceRegistry, enabled: bool) -> None:
        self.registry = registry
        self.enabled = enabled
        self.jobs: Queue[tuple[str, object]] = Queue(maxsize=4)
        self.lock = threading.RLock()
        self.worker_status = "disabled" if not enabled else "starting"
        self.worker_error: Optional[str] = None
        self.last_submitted: dict[str, float] = {}
        self.latest_embeddings: dict[str, list[float]] = {}
        self.identities: dict[str, dict] = {}
        self.active_tracks: set[str] = set()
        self.candidate_history: dict[str, deque[Optional[str]]] = defaultdict(
            lambda: deque(maxlen=max(1, FACE_CONFIRMATIONS, FACE_UNKNOWN_CONFIRMATIONS))
        )
        self.identity_event_tracks: set[str] = set()
        self.best_candidate_distances: dict[str, float] = {}
        self.last_failure_log: dict[str, float] = {}

    def start(self) -> None:
        if self.enabled:
            threading.Thread(target=self._run, daemon=True, name="face-recognition").start()

    def _run(self) -> None:
        try:
            encoder = DeepFaceEncoder(
                model_name=FACE_MODEL_NAME,
                detector_backend=FACE_DETECTOR_BACKEND,
            )
        except Exception as error:
            with self.lock:
                self.worker_status = "error"
                self.worker_error = str(error)
            print("face recognition unavailable:", error)
            return

        with self.lock:
            self.worker_status = "ready"
        print(
            "face recognition ready:",
            {"model": FACE_MODEL_NAME, "detector": FACE_DETECTOR_BACKEND},
        )

        while True:
            try:
                track_id, person_crop = self.jobs.get(timeout=1.0)
            except Empty:
                continue

            try:
                embedding = encoder.encode(person_crop)
                self._process_embedding(track_id, embedding)
            except Exception as error:
                now = time.monotonic()
                with self.lock:
                    last_log = self.last_failure_log.get(track_id, 0.0)
                    if now - last_log >= 10.0:
                        print(f"face not usable for track {track_id}:", error)
                        self.last_failure_log[track_id] = now
            finally:
                self.jobs.task_done()

    def _process_embedding(self, track_id: str, embedding: list[float]) -> None:
        nearest = self.registry.nearest(embedding)
        match = (
            nearest
            if nearest is not None and nearest.distance <= self.registry.threshold
            else None
        )
        event_to_send = None

        with self.lock:
            if track_id not in self.active_tracks:
                return
            self.latest_embeddings[track_id] = embedding
            existing = self.identities.get(track_id)
            if existing and existing.get("status") == "recognized":
                return

            if nearest is not None:
                previous_best = self.best_candidate_distances.get(track_id)
                if previous_best is None or nearest.distance < previous_best:
                    self.best_candidate_distances[track_id] = nearest.distance
                    print(
                        f"face candidate for track {track_id}: "
                        f"{nearest.name} distance={nearest.distance:.4f} "
                        f"threshold={self.registry.threshold:.4f}"
                    )

            candidate_id = match.person_id if match else None
            history = self.candidate_history[track_id]
            history.append(candidate_id)

            if match is None:
                required = max(1, FACE_UNKNOWN_CONFIRMATIONS)
                if len(history) < required or any(
                    candidate is not None for candidate in list(history)[-required:]
                ):
                    return
                if existing and existing.get("status") == "unknown":
                    return
                self.identities[track_id] = pending_identity("unknown")
                event_to_send = ("person_unknown", self.identities[track_id].copy())
            else:
                required = max(1, FACE_CONFIRMATIONS)
                if len(history) < required or any(
                    candidate != candidate_id for candidate in list(history)[-required:]
                ):
                    return
                identity = {
                    "status": "recognized",
                    "person_id": match.person_id,
                    "name": match.name,
                    "distance": round(match.distance, 4),
                    "threshold": match.threshold,
                }
                self.identities[track_id] = identity
                event_to_send = ("person_recognized", identity.copy())

        if event_to_send:
            queued = send_person_event(
                event_to_send[0],
                track_id,
                identity=event_to_send[1],
            )
            if queued:
                with self.lock:
                    if track_id in self.active_tracks:
                        self.identity_event_tracks.add(track_id)

    def submit(self, track_id: str, person_crop) -> bool:
        if not self.enabled:
            return False

        now = time.monotonic()
        with self.lock:
            self.active_tracks.add(track_id)
            if self.worker_status == "error":
                return False
            if self.identities.get(track_id, {}).get("status") == "recognized":
                return False
            if now - self.last_submitted.get(track_id, 0.0) < FACE_ANALYSIS_INTERVAL_SECONDS:
                return False
            self.last_submitted[track_id] = now

        try:
            self.jobs.put_nowait((track_id, person_crop.copy()))
            return True
        except Full:
            return False

    def mark_active(self, track_id: str) -> None:
        with self.lock:
            self.active_tracks.add(track_id)

    def identity_for(self, track_id: str) -> dict:
        with self.lock:
            identity = self.identities.get(track_id)
            if identity:
                return identity.copy()
            if not self.enabled or self.worker_status == "error":
                return pending_identity("unavailable")
            return pending_identity()

    def has_face_sample(self, track_id: str) -> bool:
        with self.lock:
            return track_id in self.latest_embeddings

    def identity_event_announced(self, track_id: str) -> bool:
        with self.lock:
            return track_id in self.identity_event_tracks

    def enroll(self, track_id: str, name: str) -> dict:
        with self.lock:
            if track_id not in self.active_tracks:
                raise LookupError("track is not active")
            embedding = self.latest_embeddings.get(track_id)
            if embedding is None:
                raise LookupError(
                    "no usable face has been captured for this active track yet"
                )

        enrolled = self.registry.enroll(name, embedding)
        identity = {
            "status": "recognized",
            "person_id": enrolled["person_id"],
            "name": enrolled["name"],
            "distance": 0.0,
            "threshold": FACE_MATCH_THRESHOLD,
        }
        with self.lock:
            self.identities[track_id] = identity

        queued = send_person_event(
            "person_enrolled",
            track_id,
            identity=identity.copy(),
        )
        if queued:
            with self.lock:
                if track_id in self.active_tracks:
                    self.identity_event_tracks.add(track_id)
        return {**enrolled, "track_id": track_id}

    def forget_track(self, track_id: str) -> None:
        with self.lock:
            self.active_tracks.discard(track_id)
            self.last_submitted.pop(track_id, None)
            self.latest_embeddings.pop(track_id, None)
            self.identities.pop(track_id, None)
            self.candidate_history.pop(track_id, None)
            self.identity_event_tracks.discard(track_id)
            self.best_candidate_distances.pop(track_id, None)
            self.last_failure_log.pop(track_id, None)

    def track_states(self) -> list[dict]:
        with self.lock:
            return [
                {"track_id": track_id, "identity": self.identity_for(track_id)}
                for track_id in sorted(self.active_tracks)
            ]

    def status(self) -> dict:
        with self.lock:
            return {
                "enabled": self.enabled,
                "state": self.worker_status,
                "error": self.worker_error,
                "model": FACE_MODEL_NAME,
                "detector": FACE_DETECTOR_BACKEND,
                "registered_people": len(self.registry.list_people()),
            }


face_registry = FaceRegistry(
    FACE_REGISTRY_PATH,
    model_name=FACE_MODEL_NAME,
    threshold=FACE_MATCH_THRESHOLD,
)
face_service = FaceRecognitionService(face_registry, FACE_RECOGNITION_ENABLED)


@app.route("/")
def index():
    with frame_lock:
        frame_ready = latest_frame is not None
    status_text = "ready" if frame_ready else "no frame"
    event_status_text = "enabled" if face_events_enabled() else "disabled"
    toggle_to = "false" if face_events_enabled() else "true"
    toggle_label = "Disable" if face_events_enabled() else "Enable"
    return Response(
        f"""<html><body>
        <h1>DeepSORT Server</h1>
        <p><a href=\"/snapshot\">Snapshot</a></p>
        <p><a href=\"/snapshot/annotated\">Annotated snapshot</a></p>
        <p><a href=\"/status\">Status</a></p>
        <p><a href=\"/faces\">Registered faces</a></p>
        <p><a href=\"/tracks\">Active face states</a></p>
        <p>Frame status: {status_text}</p>
        <p>Face/person events: <strong>{event_status_text}</strong></p>
        <button onclick=\"toggleFaceEvents()\">{toggle_label} face/person events</button>
        <script>
        async function toggleFaceEvents() {{
          await fetch('/face-events', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{enabled: {toggle_to}}})
          }});
          location.reload();
        }}
        </script>
        </body></html>""",
        mimetype="text/html",
    )


@app.route("/status")
def status():
    with frame_lock:
        frame_ready = latest_frame is not None
    response = {
        "camera": "ready" if frame_ready else "no_frame",
        "event_delivery": face_events_enabled(),
        "face_events": {
            "enabled": face_event_send_enabled,
            "effective_enabled": face_events_enabled(),
            "master_event_delivery_enabled": EVENT_SEND_ENABLED,
            "queued_events": event_queue.qsize(),
        },
        "face_recognition": face_service.status(),
    }
    return jsonify(response), 200 if frame_ready else 503


@app.route("/face-events", methods=["GET", "POST"])
def face_event_switch():
    if request.method == "GET":
        return jsonify(
            {
                "enabled": face_event_send_enabled,
                "effective_enabled": face_events_enabled(),
                "master_event_delivery_enabled": EVENT_SEND_ENABLED,
                "queued_events": event_queue.qsize(),
            }
        )

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or set(payload) != {"enabled"}:
        return jsonify({"error": "body must contain exactly enabled"}), 400
    if not isinstance(payload["enabled"], bool):
        return jsonify({"error": "enabled must be a boolean"}), 400
    return jsonify(set_face_events_enabled(payload["enabled"]))


def jpeg_response(frame) -> tuple[Response, int] | Response:
    if frame is None:
        return Response("no frame", mimetype="text/plain"), 503
    encoded, jpeg = cv2.imencode(".jpg", frame)
    if not encoded:
        return Response("encode failed", mimetype="text/plain"), 500
    return Response(jpeg.tobytes(), mimetype="image/jpeg")


@app.route("/snapshot")
def snapshot():
    with frame_lock:
        frame = None if latest_frame is None else latest_frame.copy()
    return jpeg_response(frame)


@app.route("/snapshot/annotated")
def annotated_snapshot():
    with frame_lock:
        frame = (
            None if latest_annotated_frame is None else latest_annotated_frame.copy()
        )
    return jpeg_response(frame)


@app.route("/faces", methods=["GET"])
def registered_faces():
    return jsonify({"people": face_registry.list_people()})


@app.route("/tracks", methods=["GET"])
def face_track_states():
    return jsonify({"tracks": face_service.track_states()})


@app.route("/faces/enroll", methods=["POST"])
def enroll_face():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "body must be a JSON object"}), 400
    if set(payload) != {"track_id", "name"}:
        return jsonify({"error": "body must contain exactly track_id and name"}), 400

    track_id = str(payload["track_id"]).strip()
    name = payload["name"]
    if not track_id:
        return jsonify({"error": "track_id must not be empty"}), 400
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > 80:
        return jsonify({"error": "name must be a non-empty string of at most 80 characters"}), 400
    if any(ord(character) < 32 for character in name):
        return jsonify({"error": "name must not contain control characters"}), 400

    try:
        enrolled = face_service.enroll(track_id, name.strip())
    except LookupError as error:
        return jsonify({"error": str(error)}), 409
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    return jsonify({"status": "enrolled", "person": enrolled}), 201


def run_server() -> None:
    app.run(host=HTTP_HOST, port=HTTP_PORT, debug=False, use_reloader=False)


threading.Thread(target=run_server, daemon=True, name="vision-http").start()

model = YOLO(YOLO_MODEL_PATH)
tracker = DeepSort(max_age=30)
face_service.start()
cap = cv2.VideoCapture(CAMERA_INDEX)
prev_active_ids: set[str] = set()
track_positions: dict[str, str] = {}
track_first_seen: dict[str, float] = {}
announced_presence_ids: set[str] = set()


def clamp_bbox(bounds, frame_shape) -> tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    left, top, right, bottom = (int(value) for value in bounds)
    return (
        max(0, min(width, left)),
        max(0, min(height, top)),
        max(0, min(width, right)),
        max(0, min(height, bottom)),
    )


def horizontal_position(left: int, right: int, frame_width: int) -> str:
    center_ratio = ((left + right) / 2) / max(1, frame_width)
    if center_ratio < 1 / 3:
        return "left"
    if center_ratio > 2 / 3:
        return "right"
    return "center"


def identity_label(identity: dict) -> str:
    if identity.get("status") == "recognized":
        return str(identity["name"])
    if identity.get("status") == "unknown":
        return "Unknown"
    if identity.get("status") == "unavailable":
        return "Face unavailable"
    return "Identifying"


try:
    while True:
        captured, frame = cap.read()
        if not captured:
            print("camera capture failed")
            break

        raw_snapshot = frame.copy()
        results = model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = model.names[class_id]
            if label != "person":
                continue
            detections.append(
                ([x1, y1, x2 - x1, y2 - y1], confidence, label)
            )

        tracks = tracker.update_tracks(detections, frame=frame)
        current_active_ids: set[str] = set()

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = str(track.track_id)
            current_active_ids.add(track_id)
            face_service.mark_active(track_id)
            bounds = track.to_ltrb(orig=True, orig_strict=False)
            if bounds is None:
                continue
            left, top, right, bottom = clamp_bbox(bounds, frame.shape)
            position = horizontal_position(left, right, frame.shape[1])
            track_positions[track_id] = position
            identity = face_service.identity_for(track_id)

            if (
                track.time_since_update == 0
                and right - left >= FACE_MIN_PERSON_WIDTH
                and bottom - top >= FACE_MIN_PERSON_HEIGHT
            ):
                person_crop = raw_snapshot[top:bottom, left:right]
                if person_crop.size:
                    face_service.submit(track_id, person_crop)

            color = (0, 200, 0) if identity.get("status") == "recognized" else (0, 200, 255)
            cv2.putText(
                frame,
                f"ID {track_id}: {identity_label(identity)}",
                (left, max(20, top - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
            )
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        now = time.monotonic()
        for appeared_id in current_active_ids - prev_active_ids:
            track_first_seen[appeared_id] = now

        for active_id in current_active_ids:
            if active_id in announced_presence_ids:
                continue
            if face_service.identity_event_announced(active_id):
                continue

            visible_seconds = now - track_first_seen.get(active_id, now)
            waiting_for_identity = face_service.has_face_sample(active_id)
            announce_after = (
                TRACK_IDENTITY_WAIT_SECONDS
                if waiting_for_identity
                else TRACK_EVENT_STABILITY_SECONDS
            )
            if visible_seconds < announce_after:
                continue

            queued = send_person_event(
                "person_appeared",
                active_id,
                position=track_positions.get(active_id),
                identity=face_service.identity_for(active_id),
            )
            if queued:
                announced_presence_ids.add(active_id)

        for disappeared_id in prev_active_ids - current_active_ids:
            if (
                disappeared_id in announced_presence_ids
                or face_service.identity_event_announced(disappeared_id)
            ):
                send_person_event(
                    "person_disappeared",
                    disappeared_id,
                    position=track_positions.get(disappeared_id),
                    identity=face_service.identity_for(disappeared_id),
                )
            face_service.forget_track(disappeared_id)
            track_positions.pop(disappeared_id, None)
            track_first_seen.pop(disappeared_id, None)
            announced_presence_ids.discard(disappeared_id)

        prev_active_ids = current_active_ids

        with frame_lock:
            latest_frame = raw_snapshot
            latest_annotated_frame = frame.copy()

        if DISPLAY_ENABLED:
            event_label = "ON" if face_events_enabled() else "OFF"
            event_color = (0, 200, 0) if face_events_enabled() else (0, 0, 255)
            cv2.putText(
                frame,
                f"Face events: {event_label} (E to toggle)",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                event_color,
                2,
            )
            cv2.imshow("DeepSORT", frame)
            pressed_key = cv2.waitKey(1) & 0xFF
            if pressed_key == 27:
                break
            if pressed_key in (ord("e"), ord("E")):
                set_face_events_enabled(not face_events_enabled())
finally:
    cap.release()
    if DISPLAY_ENABLED:
        cv2.destroyAllWindows()
