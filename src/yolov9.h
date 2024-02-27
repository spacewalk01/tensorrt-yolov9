#pragma once

#include "NvInfer.h"
#include <opencv2/opencv.hpp>

using namespace nvinfer1;
using namespace std;
using namespace cv;

struct Detection
{
    float confidence;
    int class_id;
    Rect box;
};

class Yolov9
{

public:

    Yolov9(string engine_path);
    
    ~Yolov9();
    
    void predict(Mat& image, vector<Detection>& boxes);

    void draw(Mat& image, const vector<Detection>& boxes);

private:

    size_t get_dim(const Dims& dims);

    void postprocess(vector<Detection>& boxes);

private:

    vector<Scalar> colors;
    vector<float*> gpu_buffers;          //!< The vector of device buffers needed for engine execution
    vector<float*> cpu_buffers;
    
    int input_h;
    int input_w;
    int num_output_boxes;
    int output_size;
    const int MAX_INPUT_SIZE = 4096 * 4096;
    float conf_threshold = 0.3;
    float nms_threshold = 0.4;
    
    cudaStream_t cuda_stream;
    IRuntime* runtime;                 //!< The TensorRT runtime used to deserialize the engine
    ICudaEngine* engine;               //!< The TensorRT engine used to run the network
    IExecutionContext* context;        //!< The context for executing inference using an ICudaEngine
};