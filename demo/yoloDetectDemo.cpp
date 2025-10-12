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
// ==========================================================
//                     *** 修正点 2.1 ***
//     包含这个头文件以访问 OpenCV Stream 的底层句柄
// ==========================================================
#include "opencv2/core/cuda_stream_accessor.hpp"

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

// --- 模型参数 ---
const std::string YOLO_TYPE         = "V5";
const int         INFER_WIDTH       = 640;
const int         INFER_HEIGHT      = 640;
const int         CLASS_NUM         = 8;
const int         LOCATE_NUM        = 4;
const int         COLOR_NUM         = 1;
const int         BBOXES_NUM        = 25200;
const double      CONFIDENCE_THRESH = 0.4;
const double      NMS_THRESH        = 0.5;

const std::vector<std::string> CLASS_NAMES = {
    "car", "watcher", "base", "armor_b2", "armor_r2", "armor_b3",
    "armor_r3", "armor_b4"
};

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


int main() {
    // 1. 加载 TensorRT 模型
    nvinfer1::IExecutionContext* armor_context = nullptr;
    std::string onnx_file = "./best_cv.onnx";
    std::string engine_file = "model.engine";

    rm::message("Loading YOLO model...", rm::MSG_NOTE);
    if (access(engine_file.c_str(), F_OK) == 0) {
        if (!rm::initTrtEngine(engine_file, &armor_context)) return -1;
    } else if (access(onnx_file.c_str(), F_OK) == 0) {
        if (!rm::initTrtOnnx(onnx_file, engine_file, &armor_context, 1U)) return -1;
    } else {
        rm::message("No model file found!", rm::MSG_ERROR);
        return -1;
    }
    rm::message("YOLO model loaded successfully.", rm::MSG_OK);

    // 2. 分配 CUDA 内存并设置 Stream
    size_t yolo_struct_size = sizeof(float) * (LOCATE_NUM + 1 + COLOR_NUM + CLASS_NUM);
    
    float* armor_output_host_buffer = nullptr;
    void* armor_output_device_buffer = nullptr;
    float* armor_input_device_buffer = nullptr;

    armor_output_host_buffer = new float[BBOXES_NUM * (yolo_struct_size / sizeof(float))];
    
    // ==========================================================
    //                     *** 修正点 1 ***
    //           使用 reinterpret_cast 进行类型转换
    // ==========================================================
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&armor_input_device_buffer), 3 * INFER_WIDTH * INFER_HEIGHT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&armor_output_device_buffer, BBOXES_NUM * yolo_struct_size));
    
    // ==========================================================
    //                     *** 修正点 2.2 ***
    //    让 OpenCV 创建 Stream，然后我们获取它的原生句柄
    // ==========================================================
    cv::cuda::Stream cv_stream;
    cudaStream_t detect_stream = cv::cuda::StreamAccessor::getStream(cv_stream);
    
    // TensorRT 的输入输出缓冲区
    void* buffers[2] = {armor_input_device_buffer, armor_output_device_buffer};
    rm::message("CUDA buffers and stream initialized.", rm::MSG_NOTE);

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
        
        cv::Mat float_cpu;
        float_gpu.download(float_cpu, cv_stream);
        cv_stream.waitForCompletion(); // 确保 download 完成

        cv::Mat blob = cv::dnn::blobFromImage(float_cpu);
        CUDA_CHECK(cudaMemcpyAsync(armor_input_device_buffer, blob.data, 3 * INFER_WIDTH * INFER_HEIGHT * sizeof(float), cudaMemcpyHostToDevice, detect_stream));

        // --- 模型推理 ---
        armor_context->enqueueV3(detect_stream);

        // --- 获取输出 ---
        detectOutput(
            armor_output_host_buffer,
            static_cast<float*>(armor_output_device_buffer),
            &detect_stream, // detectOutput 需要一个 cudaStream_t*
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
    
    // 不需要手动销毁 detect_stream，cv_stream 的析构函数会自动处理
    
    rm::message("Cleanup complete. Exiting.", rm::MSG_NOTE);
    return 0;
}