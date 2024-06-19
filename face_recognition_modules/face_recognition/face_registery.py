from face_recognition_modules.face_detection.yolov8_face import Yolov8Face
from face_recognition_modules.face_alignment.face_landmarks import FaceLandmarks
from face_recognition_modules.face_recognition.face_recognition import FaceRecognition
from face_recognition_modules.feature_search.feature_search import FeatureSearch
import os
import cv2
import numpy as np
from face_recognition_modules.database.face_registery_db import FaceRegisteryDatabse
from face_recognition_modules.configs import global_config

class FaceRegistery:
    def __init__(
        self, face_detector: Yolov8Face, landmarks_det: FaceLandmarks, face_recognition: FaceRecognition,
        feature_search: FeatureSearch, feature_search_unkown: FeatureSearch,
        load_to_search=True
        ) -> None:
        self.face_detector = face_detector
        self.landmarks_det = landmarks_det
        self.face_recognition = face_recognition
        self.feature_search = feature_search
        self.feature_search_unkown = feature_search_unkown
        self.db = FaceRegisteryDatabse()
        self.db_unknow = FaceRegisteryDatabse(unknow=True)
        if load_to_search:
            self._load()
    
    def _load(self):
        persons = self.db.get_persons()
        for person in persons:
            face_id = person[0]
            embedding = np.frombuffer(person[-1], dtype=np.float32).reshape(1, -1)
            self.feature_search.add(embedding, face_id)
        
        persons = self.db_unknow.get_persons()
        for person in persons:
            face_id = person[0]
            embedding = np.frombuffer(person[-1], dtype=np.float32).reshape(1, -1)
            self.feature_search_unkown.add(embedding, face_id)

    def register(self, image_folder, call_back=None):
        exts = ["jpg", "png", "jpeg", "bmp"]
        image_list = os.listdir(image_folder)
        for i, image_name in enumerate(image_list):
            if image_name.split(".")[-1] not in exts:
                continue
            image_path = os.path.join(image_folder, image_name)
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue
            face_box, _ = self.face_detector.run(image)
            if len(face_box) == 0:
                continue
            landmarks = self.landmarks_det.run(image, face_box[0])
            embedding = self.face_recognition.run(image, landmarks)
            person_name = image_name.split(".")[0]
            person_id = self.db.add_person(person_name, embedding)
            if call_back is not None and i % 100 == 0:
                call_back(i, len(image_list))
            self.feature_search.add(embedding, person_id)
    
    def add_person(self, person_name, embedding, to_unknow=False):
        if to_unknow:
            face_id = self.db_unknow.add_person(person_name, embedding)
            self.feature_search_unkown.add(embedding, face_id)
        else:
            face_id = self.db.add_person(person_name, embedding)
            self.feature_search.add(embedding, face_id)
    
    def get_person_name(self, person_id, from_unknow=False):
        if from_unknow:
            return self.db_unknow.get_person_name(person_id)
        else:
            return self.db.get_person_name(person_id)
    
    def clear(self):
        self.db.clear()
        self.db_unknow.clear()
        self.feature_search.clear()
        self.feature_search_unkown.clear()


if __name__ == "__main__":
    from face_recognition_modules.face_detection.yolov8_face import Yolov8Face
    from face_recognition_modules.face_alignment.face_landmarks import FaceLandmarks
    from face_recognition_modules.face_recognition.face_recognition import FaceRecognition
    
    face_detector = Yolov8Face(model_path="models/yolov8-lite-t.onnx", device="gpu")
    landmarks_det = FaceLandmarks(model_path="models/student_128.onnx", device="gpu")
    face_recognition = FaceRecognition(model_path="models/webface_r50.onnx", device="gpu")
    registery = FaceRegistery(face_detector, landmarks_det, face_recognition)
    print("start register")
    registery.register("test_images/register")
    

