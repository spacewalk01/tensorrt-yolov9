cmake_minimum_required(VERSION 3.12)
project(yolov9-tensorrt)

find_package(CUDA REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
add_definitions(-DAPI_EXPORTS)

# Include CUDA directories
include_directories(${CUDA_INCLUDE_DIRS})

# Add source files
set(SOURCES
    main.cpp
    yolov9.cpp
)

# Add CUDA source files
set(CUDA_SOURCES preprocess.cu)

# Add headers
set(HEADERS
    yolov9.h
    macros.h
    logging.h
    cuda_utils.h
    preprocess.h
)



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

# Create an executable
cuda_add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS} ${CUDA_SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBS}
)
