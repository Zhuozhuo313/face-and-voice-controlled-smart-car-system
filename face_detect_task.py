import cv2
from PySide6.QtCore import QRunnable, Signal
from PySide6.QtGui import QImage, QPixmap
from face_recognition_modules.face_detection.yolov8_face import Yolov8Face
from face_recognition_modules.face_alignment.face_landmarks import FaceLandmarks
from face_recognition_modules.face_recognition.face_recognition import FaceRecognition
from face_recognition_modules.feature_search.feature_search import FeatureSearch
from face_recognition_modules.face_recognition.face_recognition_service import FaceRecognitionService
from face_recognition_modules.face_recognition.face_registery import FaceRegistery
from face_recognition_modules.face_quality.face_quality import FaceQualityOverall
from face_recognition_modules.tracker.tracker import Tracker
from face_recognition_modules.common import crop_face_track, RecognizeRecord
from face_recognition_modules.database.face_record_db import FaceRecordDatabse
from tools.put_chinese_text import put_chinese_text
import numpy as np
from Picamera2_Img_et import Imget

face_detector = Yolov8Face(model_path="models/yolov8-lite-t.onnx", device="gpu")
landmarks_det = FaceLandmarks(model_path="models/student_128.onnx", device="gpu")
face_recognition = FaceRecognition(model_path="models/webface_r50.onnx", device="gpu")
feature_search = FeatureSearch()
feature_search_unkown = FeatureSearch()
face_registery = FaceRegistery(face_detector, landmarks_det, face_recognition, feature_search, feature_search_unkown)
getImg = Imget()

def reset_face_registery():
    face_registery.clear()

def reset_record():
    record_db = FaceRecordDatabse()
    record_db.clear()

class FaceDetectionTask(QRunnable):
    def __init__(self, signal, record_signal) -> None:
        super().__init__()
        self.signal = signal
        self.record_signal = record_signal
        self.is_stopped = False

    def stop(self):
        self.is_stopped = True
    
    def __del__(self):
        self.stop()
    
    def recognize_callback(self, record: RecognizeRecord):
        self.record_signal.emit(record)

    def run(self):
        face_quality = FaceQualityOverall()
        face_tracker = Tracker(face_detector, max_age=2, min_hits=10)
        chinese_text_drawer = put_chinese_text()      
        face_recognition_service = FaceRecognitionService(landmarks_det, face_quality, face_recognition,
                                                          feature_search, feature_search_unkown,
                                                          recognize_callback=self.recognize_callback)
        face_recognition_service.start()

        background = np.full((360, 640, 3), 114, dtype=np.uint8)
        # get first frame
        for i in range(5):
            frame = getImg.getImg()
        frame_w, frame_h = frame.shape[1], frame.shape[0]
        print(frame_w, frame_h)
        # 等比例缩放到960x540，放到背景图上，居中
        scale = min(640/frame_w, 360/frame_h)
        targe_frame_w = int(frame_w * scale)
        target_frame_h = int(frame_h * scale)
        
        while not self.is_stopped:
            # 读取视频帧
            frame = getImg.getImg()
            
            tracks, rm_track_ids = face_tracker.track(frame)
            for track in tracks:
                cropped_track = crop_face_track(frame, track)
                face_recognition_service.add_track(cropped_track)
            for track_id in rm_track_ids:
                face_recognition_service.rm_track(track_id)

            # draw tracks
            for track in tracks:
                x1, y1, x2, y2 = track[:4]
                # get person name
                person_name = face_recognition_service.get_track_name(track[4])
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                if person_name!="" and person_name!="Unkown":
                    chinese_text_drawer.draw_text(frame, (int(x1), int(y1)), person_name, 20, (0, 255, 0))
                    print(person_name)

            # 等比例缩放到960x540，放到背景图上，居中
            frame = cv2.resize(frame, (targe_frame_w, target_frame_h))
            background[(360-target_frame_h)//2:(360-target_frame_h)//2+target_frame_h,
                       (640-targe_frame_w)//2:(640-targe_frame_w)//2+targe_frame_w, :] = frame
            # 发送信号
            image = QImage(background.tobytes(), background.shape[1], background.shape[0], QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(image)
            self.signal.emit(pixmap)
        
        # 停止人脸识别服务
        face_recognition_service.stop()

class FaceRegisterTask(QRunnable):
    def __init__(self, image_folder, signal) -> None:
        super().__init__()
        self.image_folder = image_folder
        self.signal = signal
    
    def register_progress(self, i, total):
        msg = f"注册中 {i}/{total}"
        self.signal.emit(msg)

    def run(self):
        registery = FaceRegistery(face_detector, landmarks_det, face_recognition, 
                                       feature_search, feature_search_unkown, load_to_search=False)
        registery.register(self.image_folder, self.register_progress)
        self.signal.emit("注册完成！")
