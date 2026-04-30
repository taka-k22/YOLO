from deep_sort_realtime.deepsort_tracker import DeepSort # type: ignore
from ultralytics import YOLO # type: ignore
import cv2 # type: ignore
import threading
import requests # type: ignore
from flask import Flask, Response # type: ignore

# イベント送信ON/OFF
EVENT_SEND_ENABLED = False   # ← False にすると送信停止

app = Flask(__name__)
latest_frame = None


@app.route("/")
def index():
    global latest_frame
    status = "ready" if latest_frame is not None else "no frame"
    return Response(
        f"""<html><body>
        <h1>DeepSORT Server</h1>
        <p><a href=\"/snapshot\">Snapshot</a></p>
        <p><a href=\"/status\">Status</a></p>
        <p>Frame status: {status}</p>
        </body></html>""",
        mimetype="text/html",
    )


@app.route("/status")
def status():
    global latest_frame
    if latest_frame is None:
        return "no frame", 503
    return "ok"


@app.route("/snapshot")
def snapshot():
    global latest_frame
    if latest_frame is None:
        return "no frame", 503

    _, jpeg = cv2.imencode(".jpg", latest_frame)
    return Response(jpeg.tobytes(), mimetype="image/jpeg")


def run_server():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


threading.Thread(target=run_server, daemon=True).start()

model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)

cap = cv2.VideoCapture(0)
prev_active_ids = set()


def send_event(track_id: int, action: str = "appeared"):
    if not EVENT_SEND_ENABLED:
        return

    payload = {"event": "person", "action": action, "track_id": track_id}
    try:
        requests.post("http://localhost:3000/yolo_event", json=payload, timeout=0.2)
        print("Node送信:", payload)
    except Exception:
        pass

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_for_snapshot = frame.copy()

    results = model(frame, verbose=False)[0]

    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        if label != "person":
            continue

        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, label))

    tracks = tracker.update_tracks(detections, frame=frame)
    current_active_ids = set()

    for t in tracks:
        if not t.is_confirmed():
            continue

        track_id = t.track_id
        current_active_ids.add(track_id)
        l, t_, w, h = t.to_ltrb()

        cv2.putText(frame, f"ID {track_id}", (int(l), int(t_)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.rectangle(frame, (int(l), int(t_)),
                      (int(l+w), int(t_+h)), (0,255,0), 2)

    appeared_ids = current_active_ids - prev_active_ids
    for appeared_id in appeared_ids:
        send_event(appeared_id, action="appeared")

    disappeared_ids = prev_active_ids - current_active_ids
    for disappeared_id in disappeared_ids:
        send_event(disappeared_id, action="disappeared")

    prev_active_ids = current_active_ids

    latest_frame = frame_for_snapshot

    cv2.imshow("DeepSORT", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()