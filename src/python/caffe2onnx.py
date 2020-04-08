import coremltools
import onnxmltools

# Update your input name and path for your caffe model
proto_file = '/root/optimization/face_tensorrt/res/caffemodel/det1.prototxt'
input_caffe_path = '/root/optimization/face_tensorrt/res/caffemodel/det1.caffemodel'

# Update the output name and path for intermediate coreml model, or leave as is
output_coreml_model = 'model.mlmodel'

# Change this path to the output name and path for the onnx model
output_onnx_model = '/root/optimization/face_tensorrt/res/onnx/det1.onnx'


# Convert Caffe model to CoreML
coreml_model = coremltools.converters.caffe.convert((input_caffe_path, proto_file))

# Save CoreML model
coreml_model.save(output_coreml_model)

# Load a Core ML model
coreml_model = coremltools.utils.load_spec(output_coreml_model)

# Convert the Core ML model into ONNX
onnx_model = onnxmltools.convert_coreml(coreml_model)

# Save as protobuf
onnxmltools.utils.save_model(onnx_model, output_onnx_model)