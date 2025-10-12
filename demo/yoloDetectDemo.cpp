#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <chrono>
#include <thread>
#include <unistd.h> // For access()

// 必要的头文件
#include <dprm/dprm.h>
#include <dprm/cudatools.h>

// OpenCV
#include <opencv2/opencv.hpp>
#include "opencv2/cudawarping.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"
#include "opencv2/dnn.hpp" // For blobFromImage

// TensorRT
#include <NvInfer.h>
#include <NvOnnxParser.h>

#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                        << " - " << cudaGetErrorString(err) << std::endl;  \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

// 使用必要的命名空间
using namespace rm;
using namespace nvinfer1;

// --- 模型参数 (请确保这些参数与您的 v5n416best.onnx 模型完全匹配!) ---
const int         INFER_WIDTH       = 416;
const int         INFER_HEIGHT      = 416;
const int         CLASS_NUM         = 14;
const int         LOCATE_NUM        = 4;
const int         COLOR_NUM         = 1; // 假设有1个颜色值
const int         BBOXES_NUM        = 10647; // 10647
const double      CONFIDENCE_THRESH = 0.4;
const double      NMS_THRESH        = 0.5;

const std::vector<std::string> CLASS_NAMES = {"B1","B2","B3","B4","B5","BHero","R1","R2","R3","R4","R5","RHero","RQS","BQS"};

// 绘制检测框函数
void draw_bboxes(cv::Mat& image, const std::vector<rm::YoloRect>& bboxes) {
    for (const auto& box : bboxes) {
        cv::rectangle(image, box.box, cv::Scalar(0, 255, 0), 2);
        std::string label = CLASS_NAMES[box.class_id] + ": " + cv::format("%.2f", box.confidence);
        int baseline;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(image,
                      cv::Point(box.box.x, box.box.y - label_size.height - baseline),
                      cv::Point(box.box.x + label_size.width, box.box.y),
                      cv::Scalar(0, 255, 0),
                      cv::FILLED);
        cv::putText(image, label,
                    cv::Point(box.box.x, box.box.y - baseline),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}


int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "用法: " << argv[0] << " <path_to_your_model.onnx>" << std::endl;
        return -1;
    }
    std::string onnx_file = argv[1];
    std::string engine_file = onnx_file;
    size_t dot_pos = engine_file.rfind(".onnx");
    if (dot_pos != std::string::npos) engine_file.replace(dot_pos, 5, ".engine");
    else engine_file += ".engine";
    
    std::cout << "ONNX model file: " << onnx_file << std::endl;
    std::cout << "TensorRT engine file: " << engine_file << std::endl;

    // 1. 加载 TensorRT 模型
    nvinfer1::IExecutionContext* armor_context = nullptr;
    rm::message("Loading YOLO model...", rm::MSG_NOTE);
    if (access(engine_file.c_str(), F_OK) == 0) {
        if (!rm::initTrtEngine(engine_file, &armor_context)) return -1;
    } else if (access(onnx_file.c_str(), F_OK) == 0) {
        if (!rm::initTrtOnnx(onnx_file, engine_file, &armor_context, 1U)) return -1;
    } else {
        rm::message("Model file not found at: " + onnx_file, rm::MSG_ERROR);
        return -1;
    }
    rm::message("YOLO model loaded successfully.", rm::MSG_OK);

    // 2. 分配 CUDA 内存并设置 Stream
    size_t yolo_struct_size = sizeof(float) * (LOCATE_NUM + 1 + COLOR_NUM + CLASS_NUM);
    
    float* armor_output_host_buffer = nullptr;
    void* armor_output_device_buffer = nullptr;
    void* armor_input_device_buffer = nullptr;

    armor_output_host_buffer = new float[BBOXES_NUM * (yolo_struct_size / sizeof(float))];
    
    CUDA_CHECK(cudaMalloc(&armor_input_device_buffer, 3 * INFER_WIDTH * INFER_HEIGHT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&armor_output_device_buffer, BBOXES_NUM * yolo_struct_size));
    
    cv::cuda::Stream cv_stream;
    cudaStream_t detect_stream = cv::cuda::StreamAccessor::getStream(cv_stream);
    
    // ====================== 新增: 绑定 TensorRT 输入输出缓冲区 ======================
    auto engine = armor_context->getEngine();
    // 假设您的模型输入张量名字是 "images"，输出是 "output0"
    // 这是YOLOv5导出的ONNX模型的标准名称。如果不是，您需要用Netron等工具查看并修改它们。
    const char* input_name = "images";
    const char* output_name = "output0";

    if (!armor_context->setInputTensorAddress(input_name, armor_input_device_buffer)) {
        rm::message("Failed to bind input tensor address.", rm::MSG_ERROR);
        return -1;
    }
    if (!armor_context->setOutputTensorAddress(output_name, armor_output_device_buffer)) {
        rm::message("Failed to bind output tensor address.", rm::MSG_ERROR);
        return -1;
    }
    rm::message("TensorRT buffers bound successfully.", rm::MSG_OK);
    // ==============================================================================


    // 3. 初始化 HIK 相机
    rm::message("Initializing HIK camera...", rm::MSG_NOTE);
    int camera_num = -1;
    if (!rm::getHikCameraNum(camera_num) || camera_num < 1) {
        rm::message("Failed to get camera or no camera found.", rm::MSG_ERROR);
        return -1;
    }

    rm::Camera* camera = new rm::Camera();
    if (!rm::openHik(camera, 1, nullptr, nullptr, nullptr, 2500.0, 12.0, 200.0)) {
        rm::message("Failed to open camera 1.", rm::MSG_ERROR);
        delete camera;
        return -1;
    }
    rm::message("Camera opened successfully.", rm::MSG_OK);

    // 4. 主循环
    const std::string window_name = "YOLOv5 Detection";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 960, 720);

    cv::cuda::GpuMat gpu_frame, resized_gpu, float_gpu;
    auto frame_wait_tp = getTime();
    while (true) {
        std::shared_ptr<rm::Frame> frame = camera->buffer->pop();
        if (frame == nullptr || !frame->image || frame->image->empty()) {
            if (getDoubleOfS(frame_wait_tp, getTime()) > 2.0) {
                rm::message("Capture timeout!", rm::MSG_ERROR);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        frame_wait_tp = getTime();

        // --- 图像预处理 (在GPU上完成) ---
        gpu_frame.upload(*frame->image, cv_stream);
        cv::cuda::resize(gpu_frame, resized_gpu, cv::Size(INFER_WIDTH, INFER_HEIGHT), 0, 0, cv::INTER_LINEAR, cv_stream);
        cv::cuda::cvtColor(resized_gpu, resized_gpu, cv::COLOR_BGR2RGB, 0, cv_stream);
        resized_gpu.convertTo(float_gpu, CV_32F, 1.0/255.0, cv_stream);
        
        std::vector<cv::cuda::GpuMat> channels;
        cv::cuda::split(float_gpu, channels, cv_stream);
        CUDA_CHECK(cudaMemcpyAsync(armor_input_device_buffer, channels[0].data, INFER_WIDTH * INFER_HEIGHT * sizeof(float), cudaMemcpyDeviceToDevice, detect_stream));
        CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(armor_input_device_buffer) + INFER_WIDTH * INFER_HEIGHT * sizeof(float), channels[1].data, INFER_WIDTH * INFER_HEIGHT * sizeof(float), cudaMemcpyDeviceToDevice, detect_stream));
        CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(armor_input_device_buffer) + 2 * INFER_WIDTH * INFER_HEIGHT * sizeof(float), channels[2].data, INFER_WIDTH * INFER_HEIGHT * sizeof(float), cudaMemcpyDeviceToDevice, detect_stream));

        // --- 模型推理 ---
        if (!armor_context->enqueueV3(detect_stream)) {
            rm::message("TensorRT enqueueV3 failed!", rm::MSG_ERROR);
            break;
        }

        // --- 获取输出 ---
        detectOutput(
            armor_output_host_buffer,
            armor_output_device_buffer, // 这里传递 void* 就可以
            &detect_stream,
            yolo_struct_size,
            BBOXES_NUM
        );
        
        // --- NMS 后处理 ---
        frame->yolo_list = yoloArmorNMS_V5(
            armor_output_host_buffer, BBOXES_NUM, CLASS_NUM,
            CONFIDENCE_THRESH, NMS_THRESH, frame->width, frame->height,
            INFER_WIDTH, INFER_HEIGHT
        );
        
        // --- 绘制和显示 ---
        if (!frame->yolo_list.empty()) {
            draw_bboxes(*frame->image, frame->yolo_list);
        }
        cv::imshow(window_name, *frame->image);
        
        if (cv::waitKey(1) == 27) {
            rm::message("User requested exit.", rm::MSG_WARNING);
            break;
        }
    }

    // 5. 释放资源
    rm::message("Releasing resources...", rm::MSG_NOTE);
    cv::destroyAllWindows();
    delete camera;
    
    CUDA_CHECK(cudaFree(armor_input_device_buffer));
    CUDA_CHECK(cudaFree(armor_output_device_buffer));
    delete[] armor_output_host_buffer;
    
    // TensorRT 上下文和引擎由 rm 库管理，假设其析构函数会处理
    
    rm::message("Cleanup complete. Exiting.", rm::MSG_NOTE);
    return 0;
}