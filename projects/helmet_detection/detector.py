import os
import time
import threading
from datetime import datetime
import urllib.request

import cv2
import numpy as np

from core.base_project import BaseAIProject, Detection, StandardOutput
from core.device import select_torch_device


class HelmetDetector(BaseAIProject):
    """
    AI-модуль для обнаружения отсутствия касок на людях.
    """
    project_name = "Детектор касок"
    project_type = "helmet_detection"

    def __init__(self, config):
        super().__init__(config)
        self.lock = threading.Lock()
        self.model = None
        self.face_cascade = None

        self.today_str = ""
        self.today_faces = []
        self.processed_track_ids = set()

        labels_config = self.config.get("labels", {})
        self.HELMET_LABELS = set(labels_config.get("helmet", ["helmet", "Hardhat", "With Helmet"]))
        self.NO_HELMET_LABELS = set(labels_config.get("no_helmet", ["head", "NO-Hardhat", "Without Helmet"]))
        self.FACE_SIZE = (128, 128)

        thresholds = self.config.get("face_similarity_thresholds", {})
        self.MAE_THRESHOLD = thresholds.get("mae", 75.0)
        self.HIST_CORR_THRESHOLD = thresholds.get("hist_corr", 0.30)
        self.TEMPLATE_SCORE_THRESHOLD = thresholds.get("template_score", 0.25)

        # Канонические ключи конфигурации
        self.conf_threshold = self.config.get("conf_threshold", 0.5)
        self.iou_threshold = self.config.get("iou_threshold", 0.5)
        self.imgsz = self.config.get("imgsz", 960)

    def load(self):
        """Загружает модели YOLO и Haar Cascade в память."""
        try:
            from ultralytics import YOLO

            self.device = select_torch_device(self.config.get("device", "auto"))
            self.logger.info(f"[{self.project_name}] Используется устройство: {self.device}")

            model_path = self.resolve_model_path(self.config.get("model_path"))
            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(f"Файл модели не найден: {model_path}")
            self.model = YOLO(model_path)
            self.model.to(self.device)

            cascade_filename = "haarcascade_frontalface_default.xml"
            local_cascade_path = os.path.join(os.path.dirname(__file__), cascade_filename)

            if os.path.exists(local_cascade_path):
                haarcascade_path = local_cascade_path
            elif hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades') and os.path.exists(os.path.join(cv2.data.haarcascades, cascade_filename)):
                haarcascade_path = os.path.join(cv2.data.haarcascades, cascade_filename)
            else:
                self.logger.warning(f"Файл '{cascade_filename}' не найден. Попытка скачивания...")
                url = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{cascade_filename}"
                try:
                    urllib.request.urlretrieve(url, local_cascade_path)
                    haarcascade_path = local_cascade_path
                except Exception as download_error:
                    raise IOError(f"Не удалось скачать Haar-каскад: {download_error}")

            self.face_cascade = cv2.CascadeClassifier(haarcascade_path)
            if self.face_cascade.empty():
                raise IOError(f"Не удалось загрузить Haar-каскад для лиц из '{haarcascade_path}'")

            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.track(dummy_frame, persist=True, verbose=False)

            self._loaded = True
            self.logger.info(f"[{self.project_name}] Модуль успешно загружен.")

        except Exception as e:
            self.logger.exception(f"[{self.project_name}] Ошибка при загрузке модуля: {e}")
            self._loaded = False

    def process(self, frame: np.ndarray, camera_id: str, camera_name: str, frame_id: int = 0) -> StandardOutput:
        """Обрабатывает один кадр для обнаружения нарушений."""
        if not self._loaded:
            return self.error_output(camera_id, camera_name, "Модуль не загружен", frame_id)

        try:
            self._check_date()

            with self.lock:
                results = self.model.track(
                    frame, persist=True,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    imgsz=self.imgsz,
                    verbose=False,
                )

            detections, violating_persons = self._parse_results(results[0])

            alert_triggered = False
            alert_message = ""
            annotated_frame = None

            if violating_persons:
                annotated_frame = frame.copy()
                self._draw_boxes(annotated_frame, detections)

                new_violations_count = 0
                for person_detection in violating_persons:
                    if self._is_new_violation(person_detection, annotated_frame):
                        new_violations_count += 1
                        person_detection.metadata["is_new_violation"] = True
                        person_detection.metadata["face_image"] = self._extract_face(annotated_frame, person_detection.bbox)

                if new_violations_count > 0:
                    alert_triggered = True
                    alert_message = f"Обнаружено {new_violations_count} чел. без каски на '{camera_name}'"

            return StandardOutput(
                camera_id=camera_id,
                camera_name=camera_name,
                timestamp=time.time(),
                project_type=self.project_type,
                project_name=self.project_name,
                event_type="helmet_violation" if alert_triggered else "no_helmet_violation",
                detections=detections,
                summary={
                    "alert_triggered": alert_triggered,
                    "alert_message": alert_message,
                    "violating_persons_count": len(violating_persons),
                },
                frame_id=frame_id,
                frame=annotated_frame,
                level="critical" if alert_triggered else "info",
            )

        except Exception as e:
            self.logger.exception(f"[{self.project_name}] Ошибка в process: {e}")
            return self.error_output(camera_id, camera_name, str(e), frame_id)

    def unload(self):
        """Освобождает ресурсы."""
        self.model = None
        self.face_cascade = None
        self._loaded = False
        self.logger.info(f"[{self.project_name}] Модуль выгружен.")

    # --- Вспомогательные методы ---

    def _parse_results(self, result):
        """Разбирает результаты YOLO на списки объектов."""
        detections = []
        helmet_boxes = []
        violating_persons = []

        if result.boxes is None:
            return [], []

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = self.model.names[int(box.cls[0])]
            track_id = int(box.id[0]) if box.id is not None else None

            detection = Detection(class_name=label, confidence=conf, bbox=(x1, y1, x2, y2), track_id=track_id)
            detections.append(detection)

            if label in self.HELMET_LABELS:
                helmet_boxes.append((x1, y1, x2, y2))
            elif label in self.NO_HELMET_LABELS:
                violating_persons.append(detection)

        return detections, violating_persons

    def _is_new_violation(self, person_detection: Detection, frame: np.ndarray) -> bool:
        """Проверяет, является ли нарушение новым для сегодняшнего дня."""
        track_id = person_detection.track_id
        if track_id and track_id in self.processed_track_ids:
            return False

        face_img = self._extract_face(frame, person_detection.bbox)
        if face_img is None:
            return False

        for stored_face_img in self.today_faces:
            if self._are_faces_similar(stored_face_img, face_img):
                if track_id: self.processed_track_ids.add(track_id)
                return False

        self.today_faces.append(face_img)
        if track_id: self.processed_track_ids.add(track_id)

        return True

    def _extract_face(self, frame, box):
        x1, y1, x2, y2 = box
        roi = frame[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_roi, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        if len(faces) == 0: return None
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        face = gray_roi[fy : fy + fh, fx : fx + fw]
        return cv2.resize(face, self.FACE_SIZE)

    def _are_faces_similar(self, face1, face2):
        mae = np.mean(np.abs(face1.astype(np.float32) - face2.astype(np.float32)))

        hist1 = cv2.calcHist([face1], [0], None, [64], [0, 256])
        hist2 = cv2.calcHist([face2], [0], None, [64], [0, 256])
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)
        hist_corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        template_score = cv2.matchTemplate(face1, face2, cv2.TM_CCOEFF_NORMED).max()

        return mae <= self.MAE_THRESHOLD or hist_corr >= self.HIST_CORR_THRESHOLD or template_score >= self.TEMPLATE_SCORE_THRESHOLD

    def _check_date(self):
        current_day_str = datetime.now().strftime("%Y-%m-%d")
        if current_day_str != self.today_str:
            self.today_str = current_day_str
            self.today_faces.clear()
            self.processed_track_ids.clear()
            self.logger.info(f"Наступил новый день ({self.today_str}). Внутренний кэш нарушений сброшен.")

    def _draw_boxes(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = (0, 0, 255) if det.class_name in self.NO_HELMET_LABELS else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{det.class_name} {det.confidence:.1%}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)