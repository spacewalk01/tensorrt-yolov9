1. Setup [yolov9](https://github.com/WongKinYiu/yolov9) and download [yolov9-c.pt](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt) model.
3. Convert the model to onnx format:

- First, modify `utils/general.py` in yolov9 repo by following this [guide](https://github.com/WongKinYiu/yolov9/pull/20).
- Export the model
``` shell
python export.py --weights yolov9-c.pt --include onnx
```
4. Build a TensorRT engine: 

``` shell
trtexec.exe --onnx=yolov9-c.onnx --explicitBatch --saveEngine=yolov9-c.engine --fp16
```
5. Build the project: 

**Windows:**

- Set `opencv` and `tensorrt` installation paths in CMakeLists.txt
- Run:
  
``` shell
mkdir build
cd build
cmake ..
cmake --build . --config Release
```
