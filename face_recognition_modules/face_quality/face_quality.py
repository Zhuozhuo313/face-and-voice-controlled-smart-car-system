import numpy as np
import cv2


class FaceQualityOverall:
    def __init__(self, **kwargs) -> None:
        pass

    def pose_score(self, face_box: np.ndarray, landmarks: np.ndarray):
        center_x, center_y = (face_box[0] + face_box[2]) / 2, (face_box[1] + face_box[3]) / 2
        nose_x, nose_y = landmarks[2][0], landmarks[2][1]
        distance = np.sqrt((center_x - nose_x) ** 2 + (center_y - nose_y) ** 2)
        face_size = np.sqrt((face_box[2] - face_box[0]) ** 2 + (face_box[3] - face_box[1]) ** 2)
        pose_score = max(0, 1 - distance / face_size)
        return pose_score

    def sharpness_and_brightness_score(self, image: np.ndarray, face_box: np.ndarray):
        box = face_box[:]
        box = box.astype(np.int32)
        face_image = image[box[1] : box[3], box[0] : box[2], :]
        face_image_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        # blur the face image with a 5x5 guassian kernel
        blur_face_image = cv2.GaussianBlur(face_image_gray, (5, 5), sigmaX=1, sigmaY=1)
        # calculate the sharpness score
        sharpness_score = np.sum(np.abs(face_image_gray - blur_face_image)) / np.prod(face_image_gray.shape)
        sharpness_score = sharpness_score / 255.0
        sharpness_score = min(1, sharpness_score * 2)
        brightness_score = np.mean(face_image_gray)

        # normalize the brightness score
        if brightness_score < 20 or brightness_score > 230:
            brightness_score = 0
        else:
            brightness_score = 1 - abs(brightness_score - 127.5) / 127.5

        return sharpness_score, brightness_score

    def resolution_score(self, face_box: np.ndarray):
        face_width = face_box[2] - face_box[0]
        face_height = face_box[3] - face_box[1]
        resolution_score = min(1, min(face_width, face_height) / 224)
        if face_height/face_width > 2.5:
            resolution_score = 0
        
        if min(face_width, face_height) < 48:
            resolution_score = 0

        return resolution_score

    def run(self, image: np.ndarray, face_box: np.ndarray, landmarks: np.ndarray):
        pose_score = self.pose_score(face_box, landmarks)
        if pose_score < 0.3:
            return 0
        sharpness_score, brightness_score = self.sharpness_and_brightness_score(image, face_box)
        if sharpness_score<0.1:
            return 0
        resolution_score = self.resolution_score(face_box)
        if resolution_score < 48/224:
            return 0

        output = np.array([pose_score, sharpness_score, brightness_score, resolution_score])
        weight = np.array([0.3, 0.4, 0.1, 0.2])
        return np.sum(output * weight)


if __name__ == "__main__":
    from face_recognition_modules.face_alignment.face_landmarks import FaceLandmarks
    from face_recognition_modules.face_detection.yolov8_face import Yolov8Face
    import cv2

    yolo8face = Yolov8Face(model_path="models/yolov8-lite-t.onnx", device="gpu")
    landmarks_det = FaceLandmarks(model_path="models/student_128.onnx", device="gpu")
    image = cv2.imread("test_images/1.jpg")
    if image is None:
        raise Exception("read image failed")
    face_box, _ = yolo8face.run(image)
    landmarks = landmarks_det.run(image, face_box[0])
    face_quality = FaceQualityOverall()
    quality = face_quality.run(image, face_box[0], landmarks)
    print(quality)

