import cv2
from ultralytics import YOLO
from IPython.display import display, clear_output
import PIL.Image

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            print(f"{label}: {conf:.2f}")

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    display(PIL.Image.fromarray(img))
    clear_output(wait=True)
