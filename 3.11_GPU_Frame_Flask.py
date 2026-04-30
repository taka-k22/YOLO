import cv2  # type: ignore
import time
import threading
import requests  # type: ignore
from flask import Flask, Response  # type: ignore
from ultralytics import YOLO  # type: ignore

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
        <h1>YOLO Server</h1>
        <p><a href=\"/snapshot\">Snapshot</a></p>
        <p><a href=\"/status\">Status</a></p>
        <p>Frame status: {status}</p>
        </body></html>""",
        mimetype="text/html")


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
    return Response(jpeg.tobytes(),
                    mimetype="image/jpeg")


def run_server():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


threading.Thread(target=run_server, daemon=True).start()

# --- YOLO本体 ---
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
prev = time.time()

prev_state = set()


def send_event(label, action: str = "appeared"):
    # 限定: 'person' の出現/消滅のみ外部エンドポイントへ送信
    if not EVENT_SEND_ENABLED:
        return
    if str(label).lower() != "person":
        return
    payload = {"event": label, "action": action}
    try:
        requests.post("http://localhost:3000/yolo_event",
                      json=payload,
                      timeout=0.2)
        print("Node送信:", payload)
    except Exception:
        pass


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, imgsz=960, device=0, verbose=False)

    current_state = set()

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]

            current_state.add(label)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label} {conf:.2f}"
            cv2.putText(frame, text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

    #  イベント検出（新しく現れた物体）
    appeared = current_state - prev_state

    for label in appeared:
        send_event(label)

    #  イベント検出（消滅した物体）
    disappeared = prev_state - current_state
    for label in disappeared:
        send_event(label, action="disappeared")

    prev_state = current_state

    now = time.time()
    fps = 1/(now-prev)
    prev = now

    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 255), 2)

    latest_frame = frame.copy()

    cv2.imshow("YOLO 1080p", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
