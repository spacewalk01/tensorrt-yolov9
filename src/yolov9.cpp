#include "yolov9.h"
#include "logging.h"
#include "cuda_utils.h"
#include "macros.h"
#include "preprocess.h"

#include <fstream>
#include <iostream>
#include <random>

static Logger gLogger;

const vector<string> coconame = {
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"};

Yolov9::Yolov9(string engine_path)
{
    // Read the engine file
    ifstream engineStream(engine_path, ios::binary);
    engineStream.seekg(0, ios::end);
    const size_t modelSize = engineStream.tellg();
    engineStream.seekg(0, ios::beg);
    unique_ptr<char[]> engineData(new char[modelSize]);
    engineStream.read(engineData.get(), modelSize);
    engineStream.close();

    // Deserialize the tensorrt engine
    runtime = createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(engineData.get(), modelSize);
    context = engine->createExecutionContext();

    // Get input and output sizes of the model
    model_input_h = engine->getBindingDimensions(0).d[2];
    model_input_w = engine->getBindingDimensions(0).d[3];
    detection_attribute_size = engine->getBindingDimensions(1).d[1];
    num_detections = engine->getBindingDimensions(1).d[2];
    num_classes = detection_attribute_size - 4;

    // Initialize input buffers
    cpu_output_buffer = new float[detection_attribute_size * num_detections];
    CUDA_CHECK(cudaMalloc(&gpu_buffers[0], 3 * model_input_w * model_input_h * sizeof(float)));
    // Initialize output buffer
    CUDA_CHECK(cudaMalloc(&gpu_buffers[1], detection_attribute_size * num_detections * sizeof(float)));

    cuda_preprocess_init(MAX_IMAGE_SIZE);

    CUDA_CHECK(cudaStreamCreate(&cuda_stream));

    // Create random colors
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dis(100, 255);
    for (int i = 0; i < coconame.size(); i++)
    {
        Scalar color = Scalar(dis(gen), dis(gen), dis(gen));
        colors.push_back(color);
    }
}

Yolov9::~Yolov9()
{
    // Release stream and buffers
    cudaStreamDestroy(cuda_stream);
    for (int i = 0; i < 2; i++)
        CUDA_CHECK(cudaFree(gpu_buffers[i]));
    delete[] cpu_output_buffer;

    // Destroy the engine
    cuda_preprocess_destroy();
    delete context;
    delete engine;
    delete runtime;
}

//!
//! \brief Runs the TensorRT inference engine for YOLOv9
//!
void Yolov9::predict(Mat& image, vector<Detection> &output)
{
    // Preprocessing data on gpu
    cuda_preprocess(image.ptr(), image.cols, image.rows, gpu_buffers[0], model_input_w, model_input_h, cuda_stream);

    // Perform inference
    context->enqueueV2((void**)gpu_buffers, cuda_stream, nullptr);
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));

    // Memcpy from device output buffer to host output buffer
    CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer, gpu_buffers[1], num_detections * detection_attribute_size * sizeof(float), cudaMemcpyDeviceToHost, cuda_stream));

    // Perform postprocessing
    postprocess(output);
}

void Yolov9::postprocess(vector<Detection>& output)
{
    vector<Rect> boxes;
    vector<int> class_ids;
    vector<float> confidences;

    const Mat det_output(detection_attribute_size, num_detections, CV_32F, cpu_output_buffer);

    for (int i = 0; i < det_output.cols; ++i) {
        const Mat classes_scores = det_output.col(i).rowRange(4, 4 + num_classes);
        Point class_id_point;
        double score;
        minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

        if (score > conf_threshold) {
            const float cx = det_output.at<float>(0, i);
            const float cy = det_output.at<float>(1, i);
            const float ow = det_output.at<float>(2, i);
            const float oh = det_output.at<float>(3, i);
            Rect box;
            box.x = static_cast<int>((cx - 0.5 * ow));
            box.y = static_cast<int>((cy - 0.5 * oh));
            box.width = static_cast<int>(ow);
            box.height = static_cast<int>(oh);

            boxes.push_back(box);
            class_ids.push_back(class_id_point.y);
            confidences.push_back(score);
        }
    }

    vector<int> nms_result;
    dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_result);

    for (int i = 0; i < nms_result.size(); i++)
    {
        Detection result;
        int idx = nms_result[i];
        result.class_id = class_ids[idx];
        result.conf = confidences[idx];
        result.bbox = boxes[idx];
        output.push_back(result);
    }
}

void Yolov9::draw(Mat& image, const vector<Detection>& output)
{
    const float ratio_h = model_input_h / (float)image.rows;
    const float ratio_w = model_input_w / (float)image.cols;

    for (int i = 0; i < output.size(); i++)
    {
        auto detection = output[i];
        auto box = detection.bbox;
        auto class_id = detection.class_id;
        auto conf = detection.conf;

        if (ratio_h > ratio_w) 
        {
            box.x = box.x / ratio_w;
            box.y = (box.y - (model_input_h - ratio_w * image.rows) / 2) / ratio_w;
            box.width = box.width / ratio_w;
            box.height = box.height / ratio_w;
        }
        else 
        {
            box.x = (box.x - (model_input_w - ratio_h * image.cols) / 2) / ratio_h;
            box.y = box.y / ratio_h;
            box.width = box.width / ratio_h;
            box.height = box.height / ratio_h;
        }
        
        rectangle(image, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), colors[class_id], 3);

        // Detection box text
        string class_string = coconame[class_id] + ' ' + to_string(conf).substr(0, 4);
        Size text_size = getTextSize(class_string, FONT_HERSHEY_DUPLEX, 1, 2, 0);
        Rect text_rect(box.x, box.y - 40, text_size.width + 10, text_size.height + 20);
        rectangle(image, text_rect, colors[class_id], FILLED);
        putText(image, class_string, Point(box.x + 5, box.y - 10), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0), 2, 0);
    }
}
