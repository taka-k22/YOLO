import cv2 # type: ignore
import time
import threading
import requests # type: ignore
from flask import Flask, Response # type: ignore
from ultralytics import YOLO # type: ignore

# --- Flaskサーバ ---
app = Flask(__name__)
latest_frame = None

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

# 🔹 追加：状態管理
prev_state = set()

def send_event(label):
    try:
        requests.post("http://localhost:3000/yolo_event",
                      json={"event": label},
                      timeout=0.2)
        print("📡 Node送信:", label)
    except:
        pass  # 落ちてもYOLO止めない

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

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            text = f"{label} {conf:.2f}"
            cv2.putText(frame, text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,255,0), 2)

    # 🔹 イベント検出（新しく現れた物体）
    appeared = current_state - prev_state

    for label in appeared:
        send_event(label)

    prev_state = current_state

    now = time.time()
    fps = 1/(now-prev)
    prev = now

    cv2.putText(frame, f"FPS: {int(fps)}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0,255,255), 2)

    latest_frame = frame.copy()

    cv2.imshow("YOLO 1080p", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
