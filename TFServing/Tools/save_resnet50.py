from tensorflow.keras.applications import ResNet50
import tf2onnx

model = ResNet50(weights='imagenet')

model.save('TFServing/weights/resnet_trt')

