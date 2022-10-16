import onnx

onnx_model = onnx.load_model('TFServing/weights/tracknet.onnx')

BATCH_SIZE = 4
inputs = onnx_model.graph.input
for input in inputs:
    dim1 = input.type.tensor_type.shape.dim[0]
    dim1.dim_value = BATCH_SIZE

model_name = "tracknet_batch4.onnx"
onnx.save_model(onnx_model, model_name)