import numpy as np 
import cv2
from easydict import EasyDict as edict
from copy import deepcopy
from face_recognition_modules.configs.global_config import crop_width_margin, crop_height_margin

def softmax(x, axis=1):
    x_exp = np.exp(x)
    # 如果是列向量，则axis=0
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s

def resize_image(srcimg, keep_ratio=True, dst_width=640, dst_height=640):
    top, left, newh, neww = 0, 0, dst_width, dst_height
    if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
        hw_scale = srcimg.shape[0] / srcimg.shape[1]
        if hw_scale > 1:
            newh, neww = dst_height, int(dst_width / hw_scale)
            img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
            left = int((dst_width - neww) * 0.5)
            img = cv2.copyMakeBorder(img, 0, 0, left, dst_width - neww - left, cv2.BORDER_CONSTANT,
                                        value=(0, 0, 0))  # add border
        else:
            newh, neww = int(dst_height * hw_scale), dst_width
            img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
            top = int((dst_height - newh) * 0.5)
            img = cv2.copyMakeBorder(img, top, dst_height - newh - top, 0, 0, cv2.BORDER_CONSTANT,
                                        value=(0, 0, 0))
    else:
        img = cv2.resize(srcimg, (dst_width, dst_height), interpolation=cv2.INTER_AREA)
    return img, newh, neww, top, left

def draw_bbox(image, boxes: np.ndarray):
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
    return image

def draw_landmark(image, landmarkss: np.ndarray):
    for landmarks in landmarkss:
        for pt in landmarks:
            cv2.circle(image, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), 1)
    return image

class KeyFrame:
    def __init__(self):
        self.image: np.ndarray = np.empty((0, 0, 0), dtype=np.uint8)
        self.quality: float = 0.
        self.landmarks: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.box: np.ndarray = np.empty((0, 0,0,0), dtype=np.float32)

class TrackInfo:
    def __init__(self, id, bbox:np.ndarray, landmarks=None, quality=0, age=1):
        self.id = id
        self.name = ""
        self.bbox = bbox
        self.age = age
        self.key_frames = []
        self.reported = False
        self.disappeared = False
    
    def update(self, current_track):
        self.bbox = current_track.bbox
        self.age += 1
        self.key_frames.append(current_track.key_frames[0])
        #sort key frames with quality
        self.key_frames.sort(key=lambda x: x.quality, reverse=True)
        #remove key frames if number of key frames > 5
        self.key_frames = self.key_frames[:5]  

def crop_face_track(image, track: np.ndarray, crop_width_margin=crop_width_margin, crop_height_margin=crop_height_margin):
    x1, y1, x2, y2 = track[:4]
    w = x2 - x1
    h = y2 - y1
    #crop width 3x height 7x
    x_margin = (crop_width_margin - 1) // 2
    y1_margin = 0.5
    y2_margin = max(0, crop_height_margin - 1)
    crop_x1 = max(0, int(x1 - x_margin * w))
    crop_y1 = max(0, int(y1 - y1_margin * h))
    crop_x2 = min(image.shape[1], int(x2 + x_margin * w))
    crop_y2 = min(image.shape[0], int(y2 + y2_margin * h))

    x1 = x1-crop_x1
    y1 = y1-crop_y1
    x2 = x2-crop_x1
    y2 = y2-crop_y1

    cropped_track = TrackInfo(track[4], np.array([x1, y1, x2, y2]))
    key_frame = KeyFrame()
    key_frame.image = deepcopy(image[crop_y1:crop_y2, crop_x1:crop_x2, :])
    key_frame.box = np.array([x1, y1, x2, y2])
    cropped_track.key_frames.append(key_frame)

    return cropped_track

class RecognizeRecord:
    def __init__(self):
        self.person_id: str = ""
        self.person_name: str = ""
        self.time: str = ""
        self.count: int = 0
        self.score: float = 0.
        self.image: np.ndarray = np.empty((0, 0, 0), dtype=np.uint8)
        self.track_id: int = 0
        self.embedding: np.ndarray = np.empty((0, 0), dtype=np.float32)
        
    