import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
from skimage import io

BATCH_SIZE = 1

data = ""

batched4 = np.repeat(np.expand_dims(np.array(data, dtype=np.float32), axis=0), BATCH_SIZE, axis=0)
input_batch = 255*np.array(batched4, dtype=np.float32)

f = open("TFServing/weights/scrfd/trt/scrfd4_engine.trt", "rb")
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


# indices = (-trt_predictions[0]).argsort()[:5]
# print("Class | Probability (out of 1)")
# print(list(zip(indices, trt_predictions[0][indices])))