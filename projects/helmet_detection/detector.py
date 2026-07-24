import os
import time
import json
import threading
from datetime import datetime
import urllib.request
import urllib.error

import cv2
import numpy as np

# В реальной системе TolagAI эти классы будут импортированы из ядра.
# Для автономного теста можно раскомментировать заглушки, но в финальной версии их быть не должно.
from core.base_project import BaseAIProject, Detection, StandardOutput


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

        # Управление состоянием для предотвращения дубликатов
        self.today_str = ""
        self.processed_track_ids = set()

        # Загрузка меток классов из конфигурации с fallback-значениями
        labels_config = self.config.get("labels", {})
        self.PERSON_LABELS = set(labels_config.get("person", ["Person", "person"]))
        self.HELMET_LABELS = set(labels_config.get("helmet", ["Hardhat", "With Helmet", "helmet"]))
        self.NO_HELMET_LABELS = set(labels_config.get("no_helmet", ["NO-Hardhat", "Without Helmet", "head"]))
        self.FACE_SIZE = (128, 128)

        # Пороги для сравнения лиц
        thresholds = self.config.get("face_similarity_thresholds", {})
        self.MAE_THRESHOLD = thresholds.get("mae", 75.0)
        self.HIST_CORR_THRESHOLD = thresholds.get("hist_corr", 0.30)
        self.TEMPLATE_SCORE_THRESHOLD = thresholds.get("template_score", 0.25)

    def load(self):
        """Загружает модели YOLO и Haar Cascade в память."""
        try:
            from ultralytics import YOLO

            # 1. Выбор устройства (CPU/GPU)
            dev = self.config.get("device", "auto")
            if dev == "auto":
                import torch
                self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            else:
                self.device = dev
            self.logger.info(f"[{self.project_name}] Используется устройство: {self.device}")

            # 2. Загрузка модели YOLO
            model_path = self.config.get("model_path")
            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(f"Файл модели не найден: {model_path}")
            self.model = YOLO(model_path)
            self.model.to(self.device)

            # 3. Загрузка каскада для распознавания лиц
            # Сначала ищем локально, потом в пакете, потом скачиваем.
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
                    self.logger.info(f"Файл успешно скачан и сохранен как '{local_cascade_path}'")
                except Exception as download_error:
                    raise IOError(f"Не удалось скачать Haar-каскад: {download_error}")

            self.face_cascade = cv2.CascadeClassifier(haarcascade_path)
            if self.face_cascade.empty():
                raise IOError(f"Не удалось загрузить Haar-каскад для лиц из '{haarcascade_path}'")

            # 4. Прогрев модели
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
            return self._create_error_output("Модуль не загружен", camera_id, camera_name, frame_id)

        try:
            # Проверка и обновление состояния при смене дня
            self._check_date()

            with self.lock:
                results = self.model.track(frame, persist=True, conf=0.35, verbose=False)

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
                        # Добавляем метаданные к детекции, чтобы сообщить движку о новом нарушении
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
            return self._create_error_output(str(e), camera_id, camera_name, frame_id)

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
            return False # Не можем проверить без лица

        # Сравниваем с лицами, уже зарегистрированными сегодня
        for stored_face_img in self.today_faces:
            if self._are_faces_similar(stored_face_img, face_img):
                if track_id: self.processed_track_ids.add(track_id)
                return False # Нашли похожее лицо, это не новое нарушение

        # Это новое нарушение, регистрируем его
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
        """Более надежное сравнение лиц с использованием нескольких метрик."""
        # 1. Mean Absolute Error (MAE)
        mae = np.mean(np.abs(face1.astype(np.float32) - face2.astype(np.float32)))

        # 2. Histogram Correlation
        hist1 = cv2.calcHist([face1], [0], None, [64], [0, 256])
        hist2 = cv2.calcHist([face2], [0], None, [64], [0, 256])
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)
        hist_corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        # 3. Template Matching Score
        template_score = cv2.matchTemplate(face1, face2, cv2.TM_CCOEFF_NORMED).max()

        return mae <= self.MAE_THRESHOLD or hist_corr >= self.HIST_CORR_THRESHOLD or template_score >= self.TEMPLATE_SCORE_THRESHOLD

    def _check_date(self):
        """Сбрасывает состояние, если наступил новый день."""
        current_day_str = datetime.now().strftime("%Y-%m-%d")
        if current_day_str != self.today_str:
            self.today_str = current_day_str
            self.today_faces.clear()
            self.processed_track_ids.clear()
            self.logger.info(f"Наступил новый день ({self.today_str}). Внутренний кэш нарушений сброшен.")

    def _create_error_output(self, message, camera_id, camera_name, frame_id):
        return StandardOutput(
            camera_id=camera_id, camera_name=camera_name, timestamp=time.time(),
            project_type=self.project_type, project_name=self.project_name,
            event_type="project_error", detections=[],
            summary={"alert_triggered": True, "alert_message": f"Error: {message}"},
            frame_id=frame_id, frame=None, level="warning"
        )

    @staticmethod
    def _get_head_region(person_box):
        x1, y1, x2, y2 = person_box
        head_height = int((y2 - y1) * 0.35)
        return (x1, y1, x2, y1 + head_height)

    @staticmethod
    def _boxes_overlap(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

    def _draw_boxes(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = (0, 0, 255) if det.class_name in self.NO_HELMET_LABELS else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{det.class_name} {det.confidence:.1%}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)