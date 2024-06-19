from face_recognition_modules.base_model.base_model import BaseModel
import numpy as np
import cv2
from face_recognition_modules.face_alignment.face_alignment import norm_crop

class FaceRecognition(BaseModel):
    def __init__(self, model_path, device='cpu', **kwargs) -> None:
        super().__init__(model_path, device, **kwargs)
        self.input_size = (112, 112)
    
    def preprocess(self, image, landmarks):
        aligned = norm_crop(image, landmarks, self.input_size[0])
        return aligned

    def run(self, image, landmarks):
        aligned = self.preprocess(image, landmarks)
        aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        aligned = aligned.transpose((2, 0, 1))
        aligned = np.expand_dims(aligned, axis=0)
        aligned = aligned.astype(np.float32)
        aligned = (aligned - 127.5) / 127.5
        output = self.inference(aligned)
        output = output[0].reshape((1, -1))
        #normalize feature
        output = output / np.linalg.norm(output, axis=1, keepdims=True)
        return output
        


