import cv2
import time
from ultralytics import YOLO

# モデル読み込み（GPU使用）
model = YOLO("yolov8n.pt")

# カメラ初期化
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# フルHD設定
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

prev = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 推論（解像度は内部で縮小 → 高速）
    results = model(frame, imgsz=960, device=0, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]

            # 枠描画
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            # ラベル
            text = f"{label} {conf:.2f}"
            cv2.putText(frame, text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,255,0), 2)

    # FPS計測
    now = time.time()
    fps = 1/(now-prev)
    prev = now

    cv2.putText(frame, f"FPS: {int(fps)}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0,255,255), 2)

    cv2.imshow("YOLO 1080p", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
