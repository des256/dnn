import onnx
import tflite2onnx

OUTPUT_PATH = "catdog.tflite"
ONNX_PATH = "../catdog.onnx"

# convert to ONNX
tflite2onnx.convert(OUTPUT_PATH, ONNX_PATH)

# remove pesky ai.onnx.ml opset domain
onnx_model = onnx.load(ONNX_PATH)
for opset in onnx_model.opset_import:
    if opset.domain == "ai.onnx.ml":
        opset.domain = ""
        opset.version = 15
onnx.save(onnx_model, ONNX_PATH)
