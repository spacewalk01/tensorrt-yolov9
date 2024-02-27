## Installation

1. Clone YOLOv9 repo and install requirements:

``` shell
git clone https://github.com/WongKinYiu/yolov9  # clone
cd yolov9
pip install -r requirements.txt  
```

3. Download [yolov9-c.pt](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt) model.
4. Convert the model to onnx format:

- Perform re-parameterization:
  
``` shell
python reparameterize.py yolov9-c.pt yolov9-c-param.pt
```

- Export the model:
  
``` shell
python export.py --weights yolov9-c-param.pt --include onnx
```

4. Build a TensorRT engine: 

``` shell
trtexec.exe --onnx=yolov9-c-param.onnx --explicitBatch --saveEngine=yolov9-c.engine --fp16
```
5. Set `opencv` and `tensorrt` installation paths in [CMakeLists.txt](https://github.com/spacewalk01/tensorrt-yolov9/blob/main/CMakeLists.txt):

```
# Find and include OpenCV
set(OpenCV_DIR "your path to OpenCV")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Set TensorRT path if not set in environment variables
set(TENSORRT_DIR "your path to TensorRT")
```

6. Build the project:
   
``` shell
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

## Tested Environment
   - TensorRT 8.6
   - CUDA 11.8
   - Windows 10
