#include "yolov9.h"
#include "logging.h"
#include "cuda_utils.h"
#include "macros.h"
#include "preprocess.h"

#include <fstream>
#include <iostream>
#include <random>

static Logger gLogger;

const vector<string> coconame = { "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush" };

Yolov9::Yolov9(string enginePath)
{
    deserializeEngine(enginePath);

    initialize();
}

Yolov9::~Yolov9()
{
    // Release stream and buffers
    cudaStreamDestroy(mCudaStream);
    for (int i = 0; i < mGpuBuffers.size(); i++)
        CUDA_CHECK(cudaFree(mGpuBuffers[i]));
    for (int i = 0; i < mCpuBuffers.size(); i++)
        delete[] mCpuBuffers[i];

    // Destroy the engine
    cuda_preprocess_destroy();
    delete mContext;
    delete mEngine;
    delete mRuntime;
}

void Yolov9::deserializeEngine(string enginePath)
{
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << enginePath << " error!" << std::endl;
    }
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char* serializedEngine = new char[size];

    file.read(serializedEngine, size);
    file.close();

    mRuntime = createInferRuntime(gLogger);

    mEngine = mRuntime->deserializeCudaEngine(serializedEngine, size);

    mContext = mEngine->createExecutionContext();

    delete[] serializedEngine;
}

void Yolov9::initialize()
{
    mGpuBuffers.resize(mEngine->getNbBindings());
    mCpuBuffers.resize(mEngine->getNbBindings());

    for (size_t i = 0; i < mEngine->getNbBindings(); ++i)
    {
        size_t binding_size = getSizeByDim(mEngine->getBindingDimensions(i));
        mBufferBindingSizes.push_back(binding_size);
        mBufferBindingBytes.push_back(binding_size * sizeof(float));

        mCpuBuffers[i] = new float[binding_size];

        cudaMalloc(&mGpuBuffers[i], mBufferBindingBytes[i]);

        if (mEngine->bindingIsInput(i))
        {
            mInputDims.push_back(mEngine->getBindingDimensions(i));
        }
        else
        {
            mOutputDims.push_back(mEngine->getBindingDimensions(i));
        }
    }

    CUDA_CHECK(cudaStreamCreate(&mCudaStream));
        
    cuda_preprocess_init(mParams.kMaxInputImageSize);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(100, 255);

    for (int i = 0; i < coconame.size(); i++)
    {
        cv::Scalar color = cv::Scalar(dis(gen),
            dis(gen),
            dis(gen));
        colors.push_back(color);
    }
}

//!
//! \brief Runs the TensorRT inference engine for YOLOv9
//!
void Yolov9::predict(Mat& inputImage, std::vector<Detection> &bboxes)
{
    const int H = mInputDims[0].d[2];
    const int W = mInputDims[0].d[3];

    // Preprocessing
    cuda_preprocess(inputImage.ptr(), inputImage.cols, inputImage.rows, mGpuBuffers[0], W, H, mCudaStream);
    CUDA_CHECK(cudaStreamSynchronize(mCudaStream));

    // Perform inference
    if (!mContext->executeV2((void**)mGpuBuffers.data()))
    {
        cout << "inference error!" << endl;
        return;
    }

    // Memcpy from device output buffers to host output buffers
    CUDA_CHECK(cudaMemcpyAsync(mCpuBuffers[1], mGpuBuffers[1], mBufferBindingBytes[1], cudaMemcpyDeviceToHost, mCudaStream));

    postprocess(bboxes);
}

void Yolov9::postprocess(std::vector<Detection>& output)
{
    std::vector<cv::Rect> boxes;
    vector<int> class_ids;
    vector<float> confidences;
    int out_rows = 84;
    int out_cols = 8400;
    const cv::Mat det_output(out_rows, out_cols, CV_32F, (float*)mCpuBuffers[1]);

    for (int i = 0; i < det_output.cols; ++i) {
        const cv::Mat classes_scores = det_output.col(i).rowRange(4, 84);
        cv::Point class_id_point;
        double score;
        cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

        if (score > 0.1) {
            const float cx = det_output.at<float>(0, i);
            const float cy = det_output.at<float>(1, i);
            const float ow = det_output.at<float>(2, i);
            const float oh = det_output.at<float>(3, i);
            cv::Rect box;
            box.x = static_cast<int>((cx - 0.5 * ow));
            box.y = static_cast<int>((cy - 0.5 * oh));
            box.width = static_cast<int>(ow);
            box.height = static_cast<int>(oh);

            boxes.push_back(box);
            class_ids.push_back(class_id_point.y);
            confidences.push_back(score);
        }
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, mParams.confThreshold, mParams.nmsThreshold, nms_result);

    for (int i = 0; i < nms_result.size(); i++)
    {
        Detection result;
        int idx = nms_result[i];
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }
}

void Yolov9::draw_bboxes(cv::Mat& frame, const std::vector<Detection>& output)
{
    const int H = mInputDims[0].d[2];
    const int W = mInputDims[0].d[3];
    const float r_h = H / (float)frame.rows;
    const float r_w = W / (float)frame.cols;

    for (int i = 0; i < output.size(); i++)
    {
        auto detection = output[i];
        auto box = detection.box;
        auto classId = detection.class_id;
        auto confidence = detection.confidence;

        if (r_h > r_w) 
        {
            box.x = box.x / r_w;
            box.y = (box.y - (H - r_w * frame.rows) / 2) / r_w;
            box.width = box.width / r_w;
            box.height = box.height / r_w;
        }
        else 
        {
            box.x = (box.x - (W - r_h * frame.cols) / 2) / r_h;
            box.y = box.y / r_h;
            box.width = box.width / r_h;
            box.height = box.height / r_h;
        }

        float xmax = box.x + box.width;
        float ymax = box.y + box.height;

        // detection box
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(100, 255);
        
        cv::rectangle(frame, cv::Point(box.x, box.y), cv::Point(xmax, ymax), colors[classId], 3);

        // Detection box text
        std::string classString = coconame[classId] + ' ' + std::to_string(confidence).substr(0, 4);
        cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);
        cv::rectangle(frame, textBox, colors[classId], cv::FILLED);
        cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
    }
}

size_t Yolov9::getSizeByDim(const Dims& dims)
{
    size_t size = 1;

    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }

    return size;
}
