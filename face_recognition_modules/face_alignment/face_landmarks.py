from face_recognition_modules.base_model.base_model import BaseModel
import numpy as np
import cv2


class FaceLandmarks(BaseModel):
    def __init__(self, model_path, device="cpu", **kwargs) -> None:
        super().__init__(model_path, device, **kwargs)
        self.input_size = 128
        self.extend = [0.2, 0.3]

    def preprocess(self, image: np.ndarray, bbox: np.ndarray):
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]

        face_size = bbox_width
        # face_size = int(max(bbox_width, bbox_height))
        face_width = (1 + 2 * self.extend[0]) * face_size
        center = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]

        ### make the box as square
        crop_bbox = np.zeros(4, dtype=np.int32)
        crop_bbox[0] = center[0] - face_width // 2
        crop_bbox[1] = center[1] - face_width // 2
        crop_bbox[2] = center[0] + face_width // 2
        crop_bbox[3] = center[1] + face_width // 2

        # limit the box in the image
        crop_bbox[0] = max(0, crop_bbox[0])
        crop_bbox[1] = max(0, crop_bbox[1])
        crop_bbox[2] = min(image.shape[1], crop_bbox[2])
        crop_bbox[3] = min(image.shape[0], crop_bbox[3])
        
        # crop
        crop_bbox = crop_bbox.astype(np.int32)
        crop_image = image[crop_bbox[1] : crop_bbox[3], crop_bbox[0] : crop_bbox[2], :]
        crop_image = cv2.resize(crop_image, (self.input_size, self.input_size))

        return crop_image, crop_bbox

    def run(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        input, crop_box = self.preprocess(image, bbox)
        input = input.astype(np.float32)
        input = input / 255.0
        input = input.transpose((2, 0, 1))
        input = np.expand_dims(input, axis=0)
        output, _ = self.inference(input)
        landmarks = np.array(output)[:98*2].reshape(-1, 2)
        landmarks = self.postprocess(landmarks, crop_box)

        #change 98 points to 5 points
        landmarks = landmarks[[96, 97, 54, 88, 92], :]
        return landmarks

    def postprocess(self, landmarks: np.ndarray, crop_box)->np.ndarray:
        h = crop_box[3] - crop_box[1]
        w = crop_box[2] - crop_box[0]

        landmarks[:, 0] = landmarks[:, 0] * w + crop_box[0]
        landmarks[:, 1] = landmarks[:, 1] * h + crop_box[1]
        return landmarks
    
if __name__=="__main__":
    from face_recognition_modules.face_detection.yolov8_face import Yolov8Face
    import cv2
    import time

    yolo8face = Yolov8Face(model_path='models/yolov8-lite-t.onnx', device='gpu')
    landmarks_det = FaceLandmarks(model_path='models/student_128.onnx', device='gpu')
    image = cv2.imdecode(np.fromfile('test_images/register/王子文.png', dtype=np.uint8), cv2.IMREAD_COLOR)
    face_box,_ = yolo8face.run(image)
    landmarks = landmarks_det.run(image, face_box[0])

    time1 = time.time()
    for i in range(10):
        landmarks = landmarks_det.run(image, face_box[0])
    time2 = time.time()
    print('avg time: ', (time2 - time1)/10)
    for i in range(landmarks.shape[0]):
        cv2.circle(image, (int(landmarks[i][0]), int(landmarks[i][1])), 1, (0, 0, 255), 2)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    