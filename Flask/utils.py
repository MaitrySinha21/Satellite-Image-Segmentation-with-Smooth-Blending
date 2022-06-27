import cv2
import pybase64
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def Decode(image):
    imgdata = pybase64.b64decode(image)
    image1 = np.asarray(bytearray(imgdata), dtype="uint8")
    image1 = cv2.imdecode(image1, cv2.IMREAD_COLOR)
    return image1
