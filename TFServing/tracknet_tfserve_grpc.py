from __future__ import print_function
import grpc
import numpy as np
import tensorflow as tf
import cv2

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc



MAX_MESSAGE_LENGTH = 20971520

def main():

    frame = cv2.imread("TFServing/image.png")
    current_frame = cv2.resize(frame, ( 360 , 640 ))
    current_frame = current_frame.astype(np.float32)

    X = np.concatenate((current_frame, current_frame, current_frame), axis=2)
    data = np.rollaxis(X, 2, 0)
    data = np.expand_dims(data, 0)
    print(data.shape)
    # data = np.array(data) / 255.0

    channel = grpc.insecure_channel(
    "localhost:8500",
    options=[
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
    ],
)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'tracknet_trt'
    request.model_spec.signature_name = 'serving_default'
    request.inputs['input_1'].CopyFrom(tf.make_tensor_proto(data))

    result = stub.Predict(request, 5.0)  # 5 secs timeout
    result = result.outputs['activation_18'].float_val
    print(result)


if __name__ == '__main__':
  main()
