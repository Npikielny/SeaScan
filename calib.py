import numpy as np, cv2
from PIL import Image
from loguru import logger

def load_calibration(calib):
    if calib is None:
        return calib
    if type(calib) == dict:
        return calib
    calib = np.load(calib, allow_pickle=True).item()
    return calib

def unwarp_image(image, calib):
    if calib is None:
        return image
    return cv2.undistort(image, calib["mtx"], calib["dist"], None, calib["newcameramtx"])

class Calibration(object):
    def __init__(self, path):
        if path is None:
            logger.warning("No calibration set!")
            self.calib = None
        else:
            self.calib = load_calibration(path)

    def unwarp_image(self, image):
        return unwarp_image(image, self.calib)
    
    def open(self, image):
        if type(image) == np.ndarray:
            return image
        else:
            img = np.asarray(Image.open(image))
            return self.unwarp_image(img)
