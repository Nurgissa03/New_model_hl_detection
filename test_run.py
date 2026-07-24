import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout = open("output_log.txt", "w", encoding="utf-8")

import cv2
from projects.helmet_detection.detector import HelmetDetector

config = {
    "model_path": "ppe_best.pt",
    "device": "auto",
    "conf_threshold": 0.35,
    "iou_threshold": 0.5,
    "imgsz": 960,
    "labels": {
        "helmet": ["helmet"],
        "no_helmet": ["head"],
    },
}

detector = HelmetDetector(config)
detector.load()
if not detector._loaded:
    print("ЗАГРУЗКА НЕ УДАЛАСЬ — смотри лог выше для деталей")

# Открой видео с камеры (0) или файл, например "test_video.mp4"
source = 0  # замени на путь к видеофайлу, если нужно
cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print("ОШИБКА: Камера не открылась!")
else:
    print("Камера открыта успешно")

frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось прочитать кадр с камеры (ret=False)")
        break

    output = detector.process(frame, camera_id="cam1", camera_name="Test Camera", frame_id=frame_id)
    if output.detections:
        for det in output.detections:
            print(f"Найдено: {det.class_name}, conf={det.confidence:.2f}")
    frame_id += 1

    display_frame = output.frame if output.frame is not None else frame
    cv2.imshow("Helmet Detection Test", display_frame)

    if output.summary.get("alert_triggered"):
        print(output.summary.get("alert_message"))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
sys.stdout.close()