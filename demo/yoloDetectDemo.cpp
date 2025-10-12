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
#include <dprm/cudatools.h> // <--- 添加这个头文件来获取 CUDA_CHECK

// OpenCV
#include <opencv2/opencv.hpp>
#include "opencv2/cudawarping.hpp" // <--- 添加这个头文件用于GPU预处理

// TensorRT
#include <NvInfer.h>
#include <NvOnnxParser.h>

// 使用必要的命名空间
using namespace rm;
using namespace nvinfer1;

// --- 模型参数 (从您的 detector_baseline_thread 代码中提取) ---
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

// ==========================================================
//                     *** 修正点 1 ***
// 将 rm::Object 替换为 rm::YoloRect，并正确解析其成员
// ==========================================================
void draw_bboxes(cv::Mat& image, const std::vector<rm::YoloRect>& bboxes) {
    for (const auto& box : bboxes) {
        // 假设 rm::YoloRect 结构体包含 cv::Rect, class_id, 和 confidence 成员
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
    // ====================================================================
    // 1. 加载 TensorRT 模型
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
    size_t yolo_struct_size = sizeof(float) * static_cast<size_t>(LOCATE_NUM + 1 + COLOR_NUM + CLASS_NUM);
    
    float* armor_output_host_buffer = nullptr;
    void* armor_output_device_buffer = nullptr;
    void* armor_input_device_buffer = nullptr;
    cudaStream_t detect_stream;

    armor_output_host_buffer = new float[BBOXES_NUM * (yolo_struct_size / sizeof(float))];

    // ==========================================================
    //                    *** 修正点 2 ***
    // 使用 CUDA_CHECK (现在已经通过头文件引入)
    // ==========================================================
    CUDA_CHECK(cudaStreamCreate(&detect_stream));
    CUDA_CHECK(cudaMalloc(&armor_input_device_buffer, 3 * INFER_WIDTH * INFER_HEIGHT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&armor_output_device_buffer, BBOXES_NUM * yolo_struct_size));
    
    void* buffers[2] = {armor_input_device_buffer, armor_output_device_buffer};
    rm::message("CUDA buffers allocated.", rm::MSG_NOTE);

    // ====================================================================
    // 3. 初始化 HIK 相机
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

    // 为GPU预处理创建GpuMat对象
    cv::cuda::GpuMat gpu_frame, resized_gpu_frame;

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

        // ==========================================================
        //                     *** 修正点 3 ***
        // 替换不存在的 rm::preprocessGpu，手动进行预处理
        // ==========================================================
        // 1. 将cv::Mat上传到GPU
        gpu_frame.upload(*frame->image, detect_stream);
        // 2. 在GPU上进行尺寸调整
        cv::cuda::resize(gpu_frame, resized_gpu_frame, cv::Size(INFER_WIDTH, INFER_HEIGHT), 0, 0, cv::INTER_LINEAR, detect_stream);
        // 3. (假设) 在GPU上进行颜色空间转换和类型转换，并将结果放入输入缓冲区
        // 注意: 这一步通常需要自定义CUDA核函数来实现最高效率的 BGR uchar -> RGB float / 255.0
        // 这里我们用一个简化的方式，如果你的DPRM库有类似功能函数最好，否则需要写一个简单的核函数
        // 作为一个可运行的demo，我们先假设 resized_gpu_frame 的指针可以直接用（尽管格式可能不完全匹配，但可以先跑起来）
        // 理想情况下应该有一个函数来完成转换：
        // convert_and_normalize_gpu(resized_gpu_frame, armor_input_device_buffer, detect_stream);
        // 此处我们直接使用其指针，需要注意数据布局可能需要调整
        // 为了使demo能编译通过，我们假设模型输入可以直接用 resized_gpu_frame 的数据。
        // 如果推理结果不正确，就需要在这里编写一个从GpuMat到float* RGB的转换核函数。
        // armor_input_device_buffer = resized_gpu_frame.data; // 这是一个简化的假设
        // 一个更健壮的方式是，把数据从GPU下载回CPU，在CPU上处理完再上传，虽然慢，但能保证正确
        cv::Mat resized_cpu;
        resized_gpu_frame.download(resized_cpu, detect_stream);
        cv::cvtColor(resized_cpu, resized_cpu, cv::COLOR_BGR2RGB);
        resized_cpu.convertTo(resized_cpu, CV_32F, 1.0/255.0);
        CUDA_CHECK(cudaMemcpyAsync(armor_input_device_buffer, resized_cpu.data, 3 * INFER_WIDTH * INFER_HEIGHT * sizeof(float), cudaMemcpyHostToDevice, detect_stream));


        // --- 模型推理 ---
        // ==========================================================
        //                     *** 修正点 4 ***
        //            将 enqueueV2 更改为 enqueueV3
        // ==========================================================
        armor_context->enqueueV3(detect_stream);


        // --- 获取输出 ---
        // ==========================================================
        //                     *** 修正点 5 ***
        //       添加 static_cast 来匹配函数签名
        // ==========================================================
        detectOutput(
            armor_output_host_buffer,
            static_cast<float*>(armor_output_device_buffer),
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

        cv::imshow(window_name, *frame->image);
        
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
    delete camera;
    
    CUDA_CHECK(cudaFree(armor_input_device_buffer));
    CUDA_CHECK(cudaFree(armor_output_device_buffer));
    delete[] armor_output_host_buffer;
    CUDA_CHECK(cudaStreamDestroy(detect_stream));
    
    rm::message("Cleanup complete. Exiting.", rm::MSG_NOTE);
    return 0;
}