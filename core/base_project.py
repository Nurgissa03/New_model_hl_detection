import os
import time
import logging


class Detection:
    def __init__(self, class_name, confidence, bbox, track_id=None):
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox
        self.track_id = track_id
        self.metadata = {}


class StandardOutput:
    def __init__(self, camera_id, camera_name, timestamp, project_type, project_name,
                 event_type, detections, summary, frame_id, frame, level):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.timestamp = timestamp
        self.project_type = project_type
        self.project_name = project_name
        self.event_type = event_type
        self.detections = detections
        self.summary = summary
        self.frame_id = frame_id
        self.frame = frame
        self.level = level


class BaseAIProject:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.INFO)
        self._loaded = False
        self.today_faces = []

    def resolve_model_path(self, path: str) -> str:
        if not path:
            return path
        if os.path.isabs(path):
            return path
        # Определяем папку проекта (там, где лежит detector.py, вызвавший этот метод)
        import sys
        module = sys.modules[self.__class__.__module__]
        project_dir = os.path.dirname(os.path.abspath(module.__file__))
        models_dir = os.path.join(project_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        # Если путь уже содержит "models/", не дублируем
        basename = os.path.basename(path)
        return os.path.join(models_dir, basename)

    def error_output(self, camera_id, camera_name, error_msg, frame_id=0) -> StandardOutput:
        return StandardOutput(
            camera_id=camera_id,
            camera_name=camera_name,
            timestamp=time.time(),
            project_type=getattr(self, "project_type", "unknown"),
            project_name=getattr(self, "project_name", "unknown"),
            event_type="project_error",
            detections=[],
            summary={"alert_triggered": True, "alert_message": f"Error: {error_msg}"},
            frame_id=frame_id,
            frame=None,
            level="warning",
        )