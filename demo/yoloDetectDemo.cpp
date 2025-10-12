#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <chrono>
#include <thread>
#include <unistd.h> // For access()

// 假设这些是您自定义库的头文件
#include <dprm.h>
#include <cudatools.h>

// OpenCV
#include <opencv2/opencv.hpp>

// TensorRT
#include <NvInfer.h>
#include <NvOnnxParser.h>

// 使用必要的命名空间
using namespace rm;
using namespace nvinfer1;

// --- 模型参数 (从您的 detector_baseline_thread 代码中提取) ---
// 假设这是YOLOv5模型的参数
const std::string YOLO_TYPE         = "V5";
const int         INFER_WIDTH       = 640;    // 推理宽度
const int         INFER_HEIGHT      = 640;    // 推理高度
const int         CLASS_NUM         = 8;      // 类别数量
const int         LOCATE_NUM        = 4;      // 定位信息数量 (box)
const int         COLOR_NUM         = 1;      // 颜色信息数量 (假设)
const int         BBOXES_NUM        = 25200;  // V5在640x640下的输出框数量
const double      CONFIDENCE_THRESH = 0.4;    // 置信度阈值
const double      NMS_THRESH        = 0.5;    // NMS阈值

// 假设的类别名称，用于绘制标签
const std::vector<std::string> CLASS_NAMES = {
    "car", "watcher", "base", "armor_b2", "armor_r2", "armor_b3",
    "armor_r3", "armor_b4"
};

// 绘制检测框的辅助函数
void draw_bboxes(cv::Mat& image, const std::vector<rm::Object>& bboxes) {
    for (const auto& box : bboxes) {
        // 绘制矩形框
        cv::rectangle(image, box.box, cv::Scalar(0, 255, 0), 2);

        // 创建标签文本
        std::string label = CLASS_NAMES[box.class_id] + ": " + cv::format("%.2f", box.confidence);
        
        // 获取文本尺寸
        int baseline;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

        // 绘制标签背景
        cv::rectangle(image, 
                      cv::Point(box.box.x, box.box.y - label_size.height - baseline),
                      cv::Point(box.box.x + label_size.width, box.box.y),
                      cv::Scalar(0, 255, 0), 
                      cv::FILLED);

        // 放置标签文本
        cv::putText(image, label, 
                    cv::Point(box.box.x, box.box.y - baseline), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}


int main() {
    // ====================================================================
    // 1. 加载 TensorRT 模型 (来自 tensor_load.cpp)
    // ====================================================================
    nvinfer1::IExecutionContext* armor_context = nullptr;
    std::string onnx_file = "./best_cv.onnx";
    std::string engine_file = "model.engine";

    rm::message("Loading YOLO model...", rm::MSG_NOTE);
    if (access(engine_file.c_str(), F_OK) == 0) {
        rm::message("Found existing engine file, loading...", rm::MSG_NOTE);
        if (!rm::initTrtEngine(engine_file, &armor_context)) {
            rm::message("Failed to load TensorRT engine.", rm::MSG_ERROR);
            return -1;
        }
    } else if (access(onnx_file.c_str(), F_OK) == 0) {
        rm::message("Engine file not found, building from ONNX...", rm::MSG_NOTE);
        if (!rm::initTrtOnnx(onnx_file, engine_file, &armor_context, 1U)) {
            rm::message("Failed to build TensorRT engine from ONNX.", rm::MSG_ERROR);
            return -1;
        }
    } else {
        rm::message("No model file found (neither .engine nor .onnx)!", rm::MSG_ERROR);
        return -1;
    }
    rm::message("YOLO model loaded successfully.", rm::MSG_OK);


    // ====================================================================
    // 2. 分配 CUDA 内存
    // ====================================================================
    // 根据 detector_baseline_thread 推断
    size_t yolo_struct_size = sizeof(float) * static_cast<size_t>(LOCATE_NUM + 1 + COLOR_NUM + CLASS_NUM);
    
    float* armor_output_host_buffer = nullptr;   // CPU输出缓冲区
    void* armor_output_device_buffer = nullptr; // GPU输出缓冲区
    void* armor_input_device_buffer = nullptr;  // GPU输入缓冲区
    cudaStream_t detect_stream;

    // 分配主机（CPU）内存
    armor_output_host_buffer = new float[BBOXES_NUM * (yolo_struct_size/sizeof(float))];

    // 创建CUDA流
    CUDA_CHECK(cudaStreamCreate(&detect_stream));

    // 分配设备（GPU）内存
    // 假设输入是三通道浮点数
    CUDA_CHECK(cudaMalloc(&armor_input_device_buffer, 3 * INFER_WIDTH * INFER_HEIGHT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&armor_output_device_buffer, BBOXES_NUM * yolo_struct_size));
    
    // 绑定输入输出缓冲区
    void* buffers[2] = {armor_input_device_buffer, armor_output_device_buffer};

    rm::message("CUDA buffers allocated.", rm::MSG_NOTE);


    // ====================================================================
    // 3. 初始化 HIK 相机 (来自 hik_lib.cpp)
    // ====================================================================
    rm::message("Initializing HIK camera...", rm::MSG_NOTE);
    int camera_num = -1;
    if (!rm::getHikCameraNum(camera_num) || camera_num < 1) {
        rm::message("Failed to get camera number or no camera found.", rm::MSG_ERROR);
        return -1;
    }
    rm::message("Found " + std::to_string(camera_num) + " camera(s).", rm::MSG_NOTE);

    rm::Camera* camera = new rm::Camera();
    float yaw, pitch, roll;
    double exposure = 2500.0, gain = 12.0, frame_rate = 200.0;

    if (!rm::openHik(camera, 1, &yaw, &pitch, &roll, exposure, gain, frame_rate)) {
        rm::message("Failed to open camera 1.", rm::MSG_ERROR);
        delete camera;
        return -1;
    }
    rm::message("Camera opened successfully.", rm::MSG_OK);


    // ====================================================================
    // 4. 主循环：捕获、推理、绘制和显示
    // ====================================================================
    const std::string window_name = "YOLOv5 Detection";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 960, 720);

    auto frame_wait_tp = getTime();
    while (true) {
        // --- 捕获帧 ---
        std::shared_ptr<rm::Frame> frame = camera->buffer->pop();
        if (frame == nullptr || !frame->image || frame->image->empty()) {
            if (getDoubleOfS(frame_wait_tp, getTime()) > 2.0) { // 2秒超时
                rm::message("Capture timeout!", rm::MSG_ERROR);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        frame_wait_tp = getTime(); // 重置超时计时器

        // --- 图像预处理 ---
        // 将图像从CPU拷贝到GPU并进行预处理 (此函数为dprm库的假设函数)
        rm::preprocessGpu(*frame->image, armor_input_device_buffer, INFER_WIDTH, INFER_HEIGHT, detect_stream);
        
        // --- 模型推理 ---
        armor_context->enqueueV2(buffers, detect_stream, nullptr);

        // --- 获取输出 ---
        // 将检测结果从GPU拷贝回CPU
        detectOutput(
            armor_output_host_buffer,
            armor_output_device_buffer,
            &detect_stream,
            yolo_struct_size,
            BBOXES_NUM
        );
        
        // --- NMS 后处理 ---
        frame->yolo_list = yoloArmorNMS_V5(
            armor_output_host_buffer,
            BBOXES_NUM,
            CLASS_NUM,
            CONFIDENCE_THRESH,
            NMS_THRESH,
            frame->width,
            frame->height,
            INFER_WIDTH,
            INFER_HEIGHT
        );
        
        // --- 绘制检测框 ---
        if (!frame->yolo_list.empty()) {
            draw_bboxes(*frame->image, frame->yolo_list);
        }

        // --- 显示图像 ---
        cv::imshow(window_name, *frame->image);
        
        // 按 ESC 键退出
        if (cv::waitKey(1) == 27) {
            rm::message("User requested exit.", rm::MSG_WARNING);
            break;
        }
    }

    // ====================================================================
    // 5. 释放资源
    // ====================================================================
    rm::message("Releasing resources...", rm::MSG_NOTE);
    
    cv::destroyAllWindows();

    // 释放相机资源
    // 假设有一个关闭相机的函数
    // rm::closeHik(camera); 
    delete camera;
    
    // 释放CUDA内存
    CUDA_CHECK(cudaFree(armor_input_device_buffer));
    CUDA_CHECK(cudaFree(armor_output_device_buffer));
    delete[] armor_output_host_buffer;
    CUDA_CHECK(cudaStreamDestroy(detect_stream));

    // 销毁TensorRT上下文和引擎
    // 假设上下文由引擎创建，销毁引擎即可
    // delete armor_context;
    
    rm::message("Cleanup complete. Exiting.", rm::MSG_NOTE);
    return 0;
}