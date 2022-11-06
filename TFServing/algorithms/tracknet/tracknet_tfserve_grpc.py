import grpc
import numpy as np
import cv2
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

MAX_MESSAGE_LENGTH = 20 * 1024 * 1024

def preprocess(frame):
    current_frame = cv2.resize(frame, ( 640 , 360 ))
    return current_frame.astype(np.float32)

def main():
    frame_1 = cv2.imread("TFServing/Assets/1.png")
    frame_2 = cv2.imread("TFServing/Assets/2.png")
    frame_3 = cv2.imread("TFServing/Assets/3.png")

    X = np.concatenate((preprocess(frame_1), preprocess(frame_2), preprocess(frame_3)), axis=2)
    data = np.rollaxis(X, 2, 0)
    # data = np.expand_dims(data, 0)
    print(data.shape)

    channel = grpc.insecure_channel(
        "localhost:8500",
        options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ]
    )
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'tracknet_trt'
    request.model_spec.signature_name = 'serving_default'
    request.inputs['input_1'].CopyFrom(tf.make_tensor_proto([data], shape=[1,9,360,640]))

    result = stub.Predict(request, 5.0)  # 5 secs timeout

    print(result)

    # result = result.outputs['activation_18'].float_val
    # print(result)


if __name__ == '__main__':
  main()
