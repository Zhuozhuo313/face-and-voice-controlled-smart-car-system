from face_recognition_modules.base_model.base_model import BaseModel
from face_recognition_modules.tracker import sort

class Tracker:
    def __init__(
        self,
        detector: BaseModel,
        max_age=1,
        min_hits=3,
        iou_threshold=0.3,
    ) -> None:
        self.detector = detector
        self.sort_tracker = sort.Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)

    def track(self, frame):
        # detect
        boxes, _ = self.detector.run(image=frame)
        # sort
        trackers, removed_trackers = self.sort_tracker.update(boxes)
        # limit track box in frame
        for i, tracker in enumerate(trackers):
            x1, y1, x2, y2 = tracker[:4]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            trackers[i][:4] = [x1, y1, x2, y2]

        return trackers, removed_trackers


if __name__ == "__main__":
    from face_recognition_modules.face_detection.yolov8_face import Yolov8Face
    import cv2
    from face_recognition_modules.face_alignment.face_landmarks import FaceLandmarks

    yolo8face = Yolov8Face(model_path="models/yolov8-lite-t.onnx", device="gpu")
    landmarks_det = FaceLandmarks(model_path="models/student_128.onnx", device="gpu")
    tracker = Tracker(yolo8face)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        tracks, _ = tracker.track(frame)
        for track in tracks:
            bbox = track[:4]
            # get landmarks
            # landmarks = landmarks_det.run(frame, bbox)
            id = int(track[4])
            bbox = [int(i) for i in bbox]
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, str(id), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # for landmark in landmarks:
            #     cv2.circle(frame, (int(landmark[0]), int(landmark[1])), 2, (0, 255, 0), -1)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
