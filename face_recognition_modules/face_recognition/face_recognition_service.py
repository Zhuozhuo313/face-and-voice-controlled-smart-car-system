import threading
import queue
from face_recognition_modules.common import TrackInfo, KeyFrame, RecognizeRecord
from face_recognition_modules.face_alignment.face_landmarks import FaceLandmarks
from face_recognition_modules.face_quality.face_quality import FaceQualityOverall
from face_recognition_modules.face_recognition.face_recognition import FaceRecognition
from face_recognition_modules.configs.global_config import upload_hits, recognition_threshold, \
    add_person_threshold, quality_threshold, register_unkonwn
from face_recognition_modules.feature_search.feature_search import FeatureSearch
from face_recognition_modules.face_recognition.face_registery import FaceRegistery
from face_recognition_modules.database.face_registery_db import FaceRegisteryDatabse
from face_recognition_modules.database.face_record_db import FaceRecordDatabse
import time
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

class FaceRecognitionService:
    def __init__(
        self,
        landmarks_det: FaceLandmarks,
        face_quality_det: FaceQualityOverall,
        face_recognition: FaceRecognition,
        face_search: FeatureSearch,
        face_search_unkonwn: FeatureSearch,
        recognize_callback=None,
    ) -> None:
        self.face_recognition = face_recognition
        self.landmarks_det = landmarks_det
        self.face_quality = face_quality_det
        self.face_search = face_search
        self.face_search_unkonwn = face_search_unkonwn
        self.tracks_queue = queue.Queue()
        self.recognize_queue = queue.Queue()
        self.tracks_dict = {}
        self.records_dict = {}
        self.running = True
        self.upload_hits = upload_hits
        self.recognition_threshold = recognition_threshold
        
        self.recognize_callback = recognize_callback     
        self.quality_thread = None
        self.recognition_thread = None
    
    def start(self):
        self.quality_thread = threading.Thread(target=self.quality_worker)
        self.recognition_thread = threading.Thread(target=self.recognition_worker)
        self.quality_thread.start()
        self.recognition_thread.start()

    def stop(self):
        self.running = False
        if self.quality_thread is not None:
            self.quality_thread.join()
        if self.recognition_thread is not None:
            self.recognition_thread.join()
    
    def __del__(self):
        self.stop()

    def get_track_name(self, track_id):
        if track_id in self.tracks_dict:
            if self.tracks_dict[track_id].reported:
                return self.tracks_dict[track_id].name
        return ""

    def add_track(self, track_info: TrackInfo):
        self.tracks_queue.put(track_info)
     
    def rm_track(self, track_id):
        if track_id in self.tracks_dict.keys():
            if self.tracks_dict[track_id].reported:
                self.tracks_dict.pop(track_id)
                self.records_dict.pop(track_id, None)
                return
            self.tracks_dict[track_id].disappeared = True
            
            if len(self.tracks_dict[track_id].key_frames) == 0:
                last_key_frame = None
                self.recognize_queue.put((track_id, last_key_frame, True))
                return

            for key_frame in self.tracks_dict[track_id].key_frames[:-1]:
                self.recognize_queue.put((track_id, key_frame, False))
            last_key_frame = self.tracks_dict[track_id].key_frames[-1]
            self.recognize_queue.put((track_id, last_key_frame, True))
        

    def quality_worker(self):
        while self.running:
            try:
                track_info: TrackInfo = self.tracks_queue.get(timeout=30)
            except Exception:
                continue
            track_id = track_info.id
            if track_id in self.tracks_dict and self.tracks_dict[track_id].reported:
                continue

            image = track_info.key_frames[0].image
            face_box = track_info.bbox
            landmarks = self.landmarks_det.run(image, face_box)
            quality = self.face_quality.run(image, face_box, landmarks)
            
            if quality < quality_threshold:
                continue

            track_info.key_frames[0].quality = quality
            track_info.key_frames[0].landmarks = landmarks
            if track_info.id in self.tracks_dict.keys():
                self.tracks_dict[track_id].update(track_info)
            else:
                self.tracks_dict[track_id] = track_info

            if self.tracks_dict[track_id].age % self.upload_hits == 0:
                for key_frame in self.tracks_dict[track_id].key_frames:
                    self.recognize_queue.put((track_id, key_frame, False))
                
                self.tracks_dict[track_id].key_frames.clear()
            

    def recognition_worker(self):
        self.record_db = FaceRecordDatabse()
        self.face_registery_db = FaceRegisteryDatabse()
        self.face_registery_db_unknow = FaceRegisteryDatabse(unknow=True)

        def search(embedding, to_unknow=False):
            searcher: FeatureSearch = self.face_search_unkonwn if to_unknow else self.face_search
            registery_db = self.face_registery_db_unknow if to_unknow else self.face_registery_db
            face_id, score = searcher.search(embedding)
            
            if score > self.recognition_threshold:
                person_id = registery_db.get_person_id(face_id)
                if person_id is None:
                    return None, None, score
                person_name = registery_db.get_person_name(person_id)
                if person_name is None:
                    return None, None, score
                return person_id, person_name, score
            
            return None, None, score

        def update_record(record: RecognizeRecord):
            person_id = record.person_id
            current_time = record.time
            count = self.record_db.get_person_record_count(person_id)
            self.record_db.update_record(person_id, current_time)
            record.count = count + 1
            if self.recognize_callback is not None:
                self.recognize_callback(record)
            return count

        while self.running:
            try:
                key_frame_item = self.recognize_queue.get(timeout=30)
            except queue.Empty:
                continue
            track_id, key_frame, is_last_key_frame = key_frame_item

            if track_id not in self.tracks_dict.keys():
                continue

            if self.tracks_dict[track_id].reported:
                continue

            if key_frame is None:
                if not track_id in self.tracks_dict:
                    continue
                recognize_record = self.records_dict[track_id]    
                person_name = recognize_record.person_name
                embedding = recognize_record.embedding
                face_id = self.face_registery_db_unknow.add_person(person_name, embedding)
                self.face_search_unkonwn.add(embedding, face_id)
                recognize_record.person_id = face_id
                update_record(recognize_record)
    
                self.tracks_dict.pop(track_id)
                self.records_dict.pop(track_id)
                continue
            
            image = key_frame.image
            landmarks = key_frame.landmarks
            embedding = self.face_recognition.run(image, landmarks)
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            recognize_record = RecognizeRecord()
            recognize_record.time = current_time
            recognize_record.image = image
            recognize_record.track_id = track_id
            recognize_record.embedding = embedding
            # compare to known person at first 
            person_id, person_name, score = search(embedding)
            if person_id is not None:
                recognize_record.person_id = person_id
                recognize_record.person_name = person_name
                recognize_record.score = score
                count = update_record(recognize_record)
                if count == 0 and score > add_person_threshold:
                    face_id = self.face_registery_db.add_person(person_name, embedding, person_id)
                    self.face_search.add(embedding, face_id)
                
                self.tracks_dict[track_id].reported = True
                self.tracks_dict[track_id].name = person_name

                continue

            # compare to unknown person
            person_id, person_name, score = search(embedding, to_unknow=True)
            if person_id is not None:
                recognize_record.person_id = person_id
                recognize_record.person_name = person_name
                recognize_record.score = score
                count = update_record(recognize_record)
                
                self.tracks_dict[track_id].reported = True
                continue

            person_name = "Unknown"
            recognize_record.person_name = person_name
            
            if is_last_key_frame:
                face_id = self.face_registery_db_unknow.add_person(person_name, embedding)
                self.face_search_unkonwn.add(embedding, face_id)
                recognize_record.person_id = face_id
                update_record(recognize_record)
                if self.recognize_callback is not None:
                    self.recognize_callback(recognize_record)

                self.tracks_dict.pop(track_id)
                self.records_dict.pop(track_id, None)
                continue
            
            recognize_record.score = score
            self.records_dict[track_id] = recognize_record
            
