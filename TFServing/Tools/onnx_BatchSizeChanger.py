import onnx

onnx_model = onnx.load_model('TFServing/weights/scrfd/onnx/scrfd.onnx')

BATCH_SIZE = 32
inputs = onnx_model.graph.input
for input in inputs:
    dim1 = input.type.tensor_type.shape.dim[0]
    dim1.dim_value = BATCH_SIZE

model_name = "TFServing/weights/scrfd/onnx/scrfd_batch32.onnx"
onnx.save_model(onnx_model, model_name)