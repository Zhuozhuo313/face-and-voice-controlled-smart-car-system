from picamera2 import Picamera2
from libcamera import controls

class Imget:
    def __init__(self):
        self.cam = Picamera2()

        self.cam.preview_configuration.main.size = (640, 360)
        self.cam.preview_configuration.main.format = "RGB888"
        self.cam.preview_configuration.controls.FrameRate = 30
        self.cam.preview_configuration.align()
        self.cam.configure("preview")
        self.cam.start()

    def getImg(self):
        frame = self.cam.capture_array()
        return frame

    def __del__(self):
        self.cam.stop()
        self.cam.close()
