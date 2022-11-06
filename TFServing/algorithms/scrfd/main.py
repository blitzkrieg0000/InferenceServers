import cv2
from scrfd import SCRFD
import numpy as np

bboxes = {}
onnx_model = ""
detector = SCRFD(onnx_model)

img = cv2.imread("TFServing/algorithms/scrfd/faces.jpeg")

def relu(self, arr): return [x * (x > 0) for x in arr]

allBboxes, kpss = detector.detect(img, threshold=0.4, input_size=(640, 640))
for i in range(allBboxes.shape[0]):
    bbox = allBboxes[i]
    x1, y1, w, h, score = bbox.astype(np.int)
    bboxes[i] = relu([x1, y1, w, h])