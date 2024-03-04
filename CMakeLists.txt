cmake_minimum_required(VERSION 3.12)
project(yolov9-tensorrt)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
include_directories(src/)

# Add source files
set(SOURCES
    main.cpp
    src/yolov9.cpp
)

# Add headers
set(HEADERS
	src/yolov9.h
	src/macros.h
	src/logging.h
	src/cuda_utils.h
)

# Find and include CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Find and include OpenCV
set(OpenCV_DIR "D:\\Utils\\opencv\\opencv49_cpu\\build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Set TensorRT path if not set in environment variables
set(TENSORRT_DIR "D:\\Utils\\TensorRT-8.6.0.12")

# Include TensorRT
include_directories(${TENSORRT_DIR}/include)
link_directories(${TENSORRT_DIR}/lib)
set(TENSORRT_LIBS nvinfer nvinfer_plugin nvparsers nvonnxparser)

# Add executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBS}
)
