# core/base_project.py
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