#pragma once

#include "NvInfer.h"
#include <opencv2/opencv.hpp>

using namespace nvinfer1;
using namespace std;
using namespace cv;

struct Detection
{
    float conf;
    int class_id;
    Rect bbox;
};

class Yolov9
{

public:

    Yolov9(string engine_path);
    
    ~Yolov9();
    
    void predict(Mat& image, vector<Detection>& output);

    void draw(Mat& image, const vector<Detection>& output);

private:

    void postprocess(vector<Detection>& output);

private:

    float* gpu_buffers[2];               //!< The vector of device buffers needed for engine execution
    float* cpu_output_buffer;

    cudaStream_t cuda_stream;
    IRuntime* runtime;                 //!< The TensorRT runtime used to deserialize the engine
    ICudaEngine* engine;               //!< The TensorRT engine used to run the network
    IExecutionContext* context;        //!< The context for executing inference using an ICudaEngine

    // Model parameters
    int model_input_w;
    int model_input_h;
    int num_detections;
    int detection_attribute_size;
    int num_classes = 80;
    const int MAX_IMAGE_SIZE = 4096 * 4096;
    float conf_threshold = 0.3f;
    float nms_threshold = 0.4f;

    vector<Scalar> colors;
};
