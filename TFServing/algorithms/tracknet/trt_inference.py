import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
from skimage import io


BATCH_SIZE = 1
n_classes = 256
width, height = 640, 360

frame_1 = cv2.imread("TFServing/1.png")
frame_2 = cv2.imread("TFServing/2.png")
frame_3 = cv2.imread("TFServing/3.png")

output_height, output_width = frame_1.shape[:-1]
canvas = frame_1

def preprocess(frame):
    output_height, output_width = frame.shape[:-1]
    current_frame = cv2.resize(frame, ( 360 , 640 ))

    current_frame = current_frame.astype(np.float32)
    return current_frame

X = np.concatenate((preprocess(frame_1), preprocess(frame_2), preprocess(frame_3)), axis=2)

data = np.rollaxis(X, 2, 0)
img = np.expand_dims(data, 0)

print("INPUT: ",data.shape)


batched4 = np.repeat(np.expand_dims(np.array(img, dtype=np.float32), axis=0), BATCH_SIZE, axis=0)
input_batch = 255*np.array(batched4, dtype=np.float32)

f = open("TFServing/weights/resnet_engine.trt", "rb")
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 
engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

output = np.empty([BATCH_SIZE, 230400, 256], dtype = np.float32) # Need to set output dtype to FP16 to enable FP16

# Allocate device memory
d_input = cuda.mem_alloc(1 * input_batch.nbytes)
d_output = cuda.mem_alloc(1 * output.nbytes)

bindings = [int(d_input), int(d_output)]

stream = cuda.Stream()

def predict(batch): # result gets copied into output
    # Transfer input data to device
    cuda.memcpy_htod_async(d_input, batch, stream)
    # Execute model
    context.execute_async_v2(bindings, stream.handle, None)
    # Transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # Syncronize threads
    stream.synchronize()
    
    return output

result = predict(input_batch)

print("OUTPUT: ",result.shape)
print(result)
pr = result[0]

pr = pr.reshape(( height, width, n_classes )).argmax( axis=2 )

# numpy.int64 -> numpy.uint8 (0-255 ya hani)
pr = pr.astype(np.uint8) 

# Çıkışı tekrar boyutlandırırız zaten heatmap olduğundan şekiller önemli bizim için
heatmap = cv2.resize(pr, (output_width, output_height ))

# Threshold
ret, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)
kernelSize = (3, 3)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
heatmap = cv2.morphologyEx(heatmap, cv2.MORPH_OPEN, kernel)

cv2.imshow('', heatmap)
cv2.waitKey(0)

circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=5, minRadius=3, maxRadius=7)
if circles is not None:
    for circle in circles[0]:
        canvas = cv2.ellipse(canvas, (int(circle[0]), int(circle[1])), (7, 7), 0, 0, 360, (255, 255, 255), 1)
					


cv2.imshow('', canvas)
cv2.waitKey(0)





# indices = (-trt_predictions[0]).argsort()[:5]
# print("Class | Probability (out of 1)")
# print(list(zip(indices, trt_predictions[0][indices])))