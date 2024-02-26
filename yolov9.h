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
    cv::Rect box;
};

struct YoloParams
{
    float confThreshold = 0.3;
    float nmsThreshold = 0.4;
    int kMaxInputImageSize = 4096 * 4096;

};

class Yolov9
{

public:

    Yolov9(string modelPath);

    void predict(Mat& inputImage, std::vector<Detection>& bboxes);

    ~Yolov9();

    void draw_bboxes(cv::Mat& testImg, const std::vector<Detection>& bboxes);

private:

    YoloParams mParams;

    void deserializeEngine(string enginePath);

    void initialize();

    size_t getSizeByDim(const Dims& dims);

    void postprocess(std::vector<Detection>& bboxes);

private:
    std::vector<cv::Scalar> colors;

    vector<Dims> mInputDims;            //!< The dimensions of the input to the network.
    vector<Dims> mOutputDims;           //!< The dimensions of the output to the network.
    vector<float*> mGpuBuffers;          //!< The vector of device buffers needed for engine execution
    vector<float*> mCpuBuffers;
    vector<size_t> mBufferBindingBytes;
    vector<size_t> mBufferBindingSizes;
    cudaStream_t mCudaStream;

    IRuntime* mRuntime;                 //!< The TensorRT runtime used to deserialize the engine
    ICudaEngine* mEngine;               //!< The TensorRT engine used to run the network
    IExecutionContext* mContext;        //!< The context for executing inference using an ICudaEngine
};