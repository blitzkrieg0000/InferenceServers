import onnx

onnx_model = onnx.load_model('TFServing/weights/tracknet/onnx/tracknet.onnx')

BATCH_SIZE = 1
inputs = onnx_model.graph.input
for input in inputs:
    dim1 = input.type.tensor_type.shape.dim[0]
    dim1.dim_value = BATCH_SIZE

model_name = "TFServing/weights/tracknet/onnx/tracknet_batch1.onnx"
onnx.save_model(onnx_model, model_name)