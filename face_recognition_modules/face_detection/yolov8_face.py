from face_recognition_modules.base_model.base_model import BaseModel
import numpy as np
import cv2
import logging
import sys
import math
from face_recognition_modules.common import softmax, resize_image
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',stream=sys.stdout)

class Yolov8Face(BaseModel):
    def __init__(self, model_path, device='cpu',**kwargs) -> None:
        super().__init__(model_path, device, **kwargs)
        self.conf_threshold = kwargs.get('conf_threshold', 0.5)
        self.iou_threshold = kwargs.get('iou_threshold', 0.4)
        self.input_size = kwargs.get('input_size', 640)
        self.input_width, self.input_height = self.input_size, self.input_size
        self.reg_max=16
        self.project = np.arange(self.reg_max)
        self.strides=[8, 16, 32]

        self.feats_hw = [(math.ceil(self.input_height / self.strides[i]), math.ceil(self.input_width / self.strides[i])) for i in range(len(self.strides))]
        self.anchors = self.make_anchors(self.feats_hw)

    def make_anchors(self, feats_hw, grid_cell_offset=0.5):
        """Generate anchors from features."""
        anchor_points = {}
        for i, stride in enumerate(self.strides):
            h,w = feats_hw[i]
            x = np.arange(0, w) + grid_cell_offset  # shift x
            y = np.arange(0, h) + grid_cell_offset  # shift y
            sx, sy = np.meshgrid(x, y)
            # sy, sx = np.meshgrid(y, x)
            anchor_points[stride] = np.stack((sx, sy), axis=-1).reshape(-1, 2)
        return anchor_points
    
    def preprocess(self, image, **kwargs):
        return resize_image(image, keep_ratio=True, dst_width=self.input_width, dst_height=self.input_height)
    
    def distance2bbox(self, points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)

    def postprocess(self, preds, scale_h, scale_w, top, left, **kwargs):
        bboxes, scores, landmarks = [], [], []
        for i, pred in enumerate(preds):
            stride = int(self.input_height/pred.shape[2])
            pred = pred.transpose((0, 2, 3, 1))
            
            box = pred[..., :self.reg_max * 4]
            cls = 1 / (1 + np.exp(-pred[..., self.reg_max * 4:-15])).reshape((-1,1))
            kpts = pred[..., -15:].reshape((-1,15)) ### x1,y1,score1, ..., x5,y5,score5

            # tmp = box.reshape(self.feats_hw[i][0], self.feats_hw[i][1], 4, self.reg_max)
            tmp = box.reshape(-1, 4, self.reg_max)
            bbox_pred = softmax(tmp, axis=-1)
            bbox_pred = np.dot(bbox_pred, self.project).reshape((-1,4))

            bbox = self.distance2bbox(self.anchors[stride], bbox_pred, max_shape=(self.input_height, self.input_width)) * stride
            kpts[:, 0::3] = (kpts[:, 0::3] * 2.0 + (self.anchors[stride][:, 0].reshape((-1,1)) - 0.5)) * stride
            kpts[:, 1::3] = (kpts[:, 1::3] * 2.0 + (self.anchors[stride][:, 1].reshape((-1,1)) - 0.5)) * stride
            kpts[:, 2::3] = 1 / (1+np.exp(-kpts[:, 2::3]))

            bbox -= np.array([[left, top, left, top]])  ###合理使用广播法则
            bbox *= np.array([[scale_w, scale_h, scale_w, scale_h]])
            kpts -= np.tile(np.array([left, top, 0]), 5).reshape((1,15))
            kpts *= np.tile(np.array([scale_w, scale_h, 1]), 5).reshape((1,15))

            bboxes.append(bbox)
            scores.append(cls)
            landmarks.append(kpts)

        bboxes = np.concatenate(bboxes, axis=0)
        scores = np.concatenate(scores, axis=0)
        landmarks = np.concatenate(landmarks, axis=0)
    
        bboxes_wh = bboxes.copy()
        bboxes_wh[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]  ####xywh
        classIds = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)  ####max_class_confidence
        
        mask = confidences>self.conf_threshold
        bboxes_wh = bboxes_wh[mask]  ###合理使用广播法则
        confidences = confidences[mask]
        classIds = classIds[mask]
        landmarks = landmarks[mask]

        if len(bboxes_wh) == 0:
            return np.empty((0, 5)), np.empty((0, 5))
        
        indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(), self.conf_threshold,
                                   self.iou_threshold).flatten()
        if len(indices) > 0:
            mlvl_bboxes = bboxes_wh[indices]
            confidences = confidences[indices]
            classIds = classIds[indices]
            ## convert box to x1,y1,x2,y2
            mlvl_bboxes[:, 2:4] = mlvl_bboxes[:, 2:4] + mlvl_bboxes[:, 0:2]

            # concat box, confidence, classId
            mlvl_bboxes = np.concatenate((mlvl_bboxes, confidences.reshape(-1, 1), classIds.reshape(-1, 1)), axis=1)
            
            landmarks = landmarks[indices]
            return mlvl_bboxes, landmarks.reshape(-1, 5, 3)[..., :2]
        else:
            return np.empty((0, 5)), np.empty((0, 5))

    
    def run(self, image, **kwargs):
        img, newh, neww, top, left = self.preprocess(image)
        scale_h, scale_w = image.shape[0]/newh, image.shape[1]/neww
        # convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        output = self.inference(img)
        bboxes, landmarks = self.postprocess(output, scale_h, scale_w, top, left)
        # limit box in image
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, image.shape[1])
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, image.shape[0])
        
        return bboxes, landmarks
    
if __name__ == "__main__":
    from face_recognition_modules.common import draw_bbox, draw_landmark
    import time
    yolo8face = Yolov8Face(model_path='models/yolov8-lite-t.onnx', device='gpu')
    image = cv2.imdecode(np.fromfile('test_images/register/1.png', dtype=np.uint8), cv2.IMREAD_COLOR)
    result = yolo8face.run(image)
    t1 = time.time()
    for i in range(10):
        result = yolo8face.run(image)
    t2 = time.time()
    print('time cost: ', (t2-t1)/10)
    image = draw_bbox(image, result[0])
    image = draw_landmark(image, result[1])
    cv2.imshow('image', image)
    cv2.waitKey(0)


