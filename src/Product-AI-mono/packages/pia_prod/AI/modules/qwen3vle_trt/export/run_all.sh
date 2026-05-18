echo "First Stage - a_export_to_onnx"
python packages/pia_prod/AI/modules/qwen3vle_trt/export/a_export_to_onnx.py

echo "Second Stage - b_export_onnx_vision"
python packages/pia_prod/AI/modules/qwen3vle_trt/export/b_export_onnx_vision.py

echo "Third Stage - c_export_onnx_to_trt"
python packages/pia_prod/AI/modules/qwen3vle_trt/export/c_export_onnx_to_trt.py

echo "FINISHED!" 