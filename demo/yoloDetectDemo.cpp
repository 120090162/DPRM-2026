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

// ====================================================================================
// 1. 参数结构体：将所有可配置参数集中管理
// ====================================================================================
struct AppParams {
    // 模型相关参数
    std::string onnx_path;
    int infer_width       = 416;
    int infer_height      = 416;
    int class_num         = 14;
    int bboxes_num        = 10647;
    double conf_thresh    = 0.4;
    double nms_thresh     = 0.5;

    // 相机相关参数
    double exposure       = 2500.0;
    double gain           = 12.0;
    double gamma          = 200.0; // 注意：DPRM库中gamma的范围通常是0-255
};

// ====================================================================================
// 2. 帮助与解析函数
// ====================================================================================
void print_usage(const char* prog_name, const AppParams& defaults) {
    std::cout << "\n用法: " << prog_name << " -m <path_to_model.onnx> [可选参数]\n\n"
              << "必需参数:\n"
              << "  -m, --model <path>      ONNX模型文件的路径。\n\n"
              << "可选模型参数:\n"
              << "  --width <int>           模型推理宽度 (默认: " << defaults.infer_width << ")\n"
              << "  --height <int>          模型推理高度 (默认: " << defaults.infer_height << ")\n"
              << "  --class-num <int>       模型类别数量 (默认: " << defaults.class_num << ")\n"
              << "  --bboxes-num <int>      模型输出的BBoxes数量 (默认: " << defaults.bboxes_num << ")\n"
              << "  --conf <float>          置信度阈值 (默认: " << defaults.conf_thresh << ")\n"
              << "  --nms <float>           NMS阈值 (默认: " << defaults.nms_thresh << ")\n\n"
              << "可选相机参数:\n"
              << "  --exposure <float>      相机曝光时间 (默认: " << defaults.exposure << ")\n"
              << "  --gain <float>          相机增益 (默认: " << defaults.gain << ")\n"
              << "  --gamma <float>         相机Gamma值 (默认: " << defaults.gamma << ")\n\n"
              << "其他:\n"
              << "  -h, --help              显示此帮助信息。\n" << std::endl;
}

void parse_arguments(int argc, char* argv[], AppParams& params) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0], AppParams());
            exit(0);
        } else if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            params.onnx_path = argv[++i];
        } else if (arg == "--width" && i + 1 < argc) {
            params.infer_width = std::stoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            params.infer_height = std::stoi(argv[++i]);
        } else if (arg == "--class-num" && i + 1 < argc) {
            params.class_num = std::stoi(argv[++i]);
        } else if (arg == "--bboxes-num" && i + 1 < argc) {
            params.bboxes_num = std::stoi(argv[++i]);
        } else if (arg == "--conf" && i + 1 < argc) {
            params.conf_thresh = std::stod(argv[++i]);
        } else if (arg == "--nms" && i + 1 < argc) {
            params.nms_thresh = std::stod(argv[++i]);
        } else if (arg == "--exposure" && i + 1 < argc) {
            params.exposure = std::stod(argv[++i]);
        } else if (arg == "--gain" && i + 1 < argc) {
            params.gain = std::stod(argv[++i]);
        } else if (arg == "--gamma" && i + 1 < argc) {
            params.gamma = std::stod(argv[++i]);
        } else {
            std::cerr << "错误: 未知或不完整的参数: " << arg << std::endl;
            print_usage(argv[0], AppParams());
            exit(1);
        }
    }
}

// 绘制检测框函数 (保持不变)
const std::vector<std::string> CLASS_NAMES = {"B1","B2","B3","B4","B5","BHero","R1","R2","R3","R4","R5","RHero","RQS","BQS"};
void draw_bboxes(cv::Mat& image, const std::vector<rm::YoloRect>& bboxes) {
    for (const auto& box : bboxes) {
        if (box.class_id >= CLASS_NAMES.size()) continue; // 安全检查
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


// ====================================================================================
// 3. 主函数：使用参数结构体
// ====================================================================================
int main(int argc, char* argv[]) {
    // 创建参数对象并解析命令行
    AppParams params;
    parse_arguments(argc, argv, params);
    
    // 检查必需的参数是否提供
    if (params.onnx_path.empty()) {
        std::cerr << "错误: 必须通过 -m 或 --model 提供模型路径。" << std::endl;
        print_usage(argv[0], AppParams());
        return -1;
    }
    
    std::string engine_file = params.onnx_path;
    size_t dot_pos = engine_file.rfind(".onnx");
    if (dot_pos != std::string::npos) engine_file.replace(dot_pos, 5, ".engine");
    else engine_file += ".engine";
    
    std::cout << "ONNX model file: " << params.onnx_path << std::endl;
    std::cout << "TensorRT engine file: " << engine_file << std::endl;

    // 1. 加载 TensorRT 模型
    nvinfer1::IExecutionContext* armor_context = nullptr;
    rm::message("Loading YOLO model...", rm::MSG_NOTE);
    if (access(engine_file.c_str(), F_OK) == 0) {
        if (!rm::initTrtEngine(engine_file, &armor_context)) return -1;
    } else if (access(params.onnx_path.c_str(), F_OK) == 0) {
        if (!rm::initTrtOnnx(params.onnx_path, engine_file, &armor_context, 1U)) return -1;
    } else {
        rm::message("Model file not found at: " + params.onnx_path, rm::MSG_ERROR);
        return -1;
    }
    rm::message("YOLO model loaded successfully.", rm::MSG_OK);

    // 2. 分配 CUDA 内存并设置 Stream
    const int LOCATE_NUM = 4; // 这些通常是固定的
    const int COLOR_NUM = 1;
    size_t yolo_struct_size = sizeof(float) * (LOCATE_NUM + 1 + COLOR_NUM + params.class_num);
    
    float* armor_output_host_buffer = nullptr;
    void* armor_output_device_buffer = nullptr;
    void* armor_input_device_buffer = nullptr;

    armor_output_host_buffer = new float[params.bboxes_num * (yolo_struct_size / sizeof(float))];
    
    CUDA_CHECK(cudaMalloc(&armor_input_device_buffer, 3 * params.infer_width * params.infer_height * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&armor_output_device_buffer, params.bboxes_num * yolo_struct_size));
    
    cv::cuda::Stream cv_stream;
    cudaStream_t detect_stream = cv::cuda::StreamAccessor::getStream(cv_stream);
    
    // 绑定 TensorRT 输入输出缓冲区
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

    // 3. 初始化 HIK 相机
    rm::message("Initializing HIK camera...", rm::MSG_NOTE);
    int camera_num = -1;
    if (!rm::getHikCameraNum(camera_num) || camera_num < 1) {
        rm::message("Failed to get camera or no camera found.", rm::MSG_ERROR);
        return -1;
    }

    rm::Camera* camera = new rm::Camera();
    // 使用 params 中的相机参数
    if (!rm::openHik(camera, 1, nullptr, nullptr, nullptr, params.exposure, params.gain, params.gamma)) {
        rm::message("Failed to open camera 1.", rm::MSG_ERROR);
        delete camera;
        return -1;
    }
    rm::message("Camera opened successfully.", rm::MSG_OK);

    // 4. 主循环
    const std::string window_name = "YOLOv5 Detection";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 960, 720);

    // [新增] 为预处理后的图像创建调试窗口
    const std::string debug_window_name = "Preprocessed Input (416x416)";
    cv::namedWindow(debug_window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(debug_window_name, 416, 416); // [修改] 使用正确的窗口名


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

        // --- 图像预处理 ---
        gpu_frame.upload(*frame->image, cv_stream);
        cv::cuda::resize(gpu_frame, resized_gpu, cv::Size(params.infer_width, params.infer_height), 0, 0, cv::INTER_LINEAR, cv_stream);
        cv::cuda::cvtColor(resized_gpu, resized_gpu, cv::COLOR_BGR2RGB, 0, cv_stream);
        resized_gpu.convertTo(float_gpu, CV_32F, 1.0/255.0, cv_stream);

        // ======================= [新增] 调试代码块开始 =======================
        {
            cv::Mat preprocessed_cpu;
            cv::cuda::GpuMat temp_gpu_for_display;

            // 1. 将浮点型数据转换回 8-bit * 255
            float_gpu.convertTo(temp_gpu_for_display, CV_8U, 255.0, cv_stream);

            // 2. 将 RGB 转换回 BGR 以便 imshow 正确显示
            cv::cuda::cvtColor(temp_gpu_for_display, temp_gpu_for_display, cv::COLOR_RGB2BGR, 0, cv_stream);

            // 3. 从 GPU 下载到 CPU
            temp_gpu_for_display.download(preprocessed_cpu, cv_stream);

            // 4. 等待 CUDA stream 完成下载操作 (非常重要，否则 preprocessed_cpu 可能数据不完整)
            cv_stream.waitForCompletion();

            // 5. 显示图像
            cv::imshow(debug_window_name, preprocessed_cpu);
        }
        // ======================= [新增] 调试代码块结束 =======================
        
        std::vector<cv::cuda::GpuMat> channels;
        cv::cuda::split(float_gpu, channels, cv_stream);
        CUDA_CHECK(cudaMemcpyAsync(armor_input_device_buffer, channels[0].data, params.infer_width * params.infer_height * sizeof(float), cudaMemcpyDeviceToDevice, detect_stream));
        CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(armor_input_device_buffer) + params.infer_width * params.infer_height * sizeof(float), channels[1].data, params.infer_width * params.infer_height * sizeof(float), cudaMemcpyDeviceToDevice, detect_stream));
        CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(armor_input_device_buffer) + 2 * params.infer_width * params.infer_height * sizeof(float), channels[2].data, params.infer_width * params.infer_height * sizeof(float), cudaMemcpyDeviceToDevice, detect_stream));

        // --- 模型推理 ---
        if (!armor_context->enqueueV3(detect_stream)) {
            rm::message("TensorRT enqueueV3 failed!", rm::MSG_ERROR);
            break;
        }

        // --- 获取输出 ---
        detectOutput(
            armor_output_host_buffer,
            static_cast<const float*>(armor_output_device_buffer), // 这里必须传递 float* ，否则会出错
            &detect_stream,
            yolo_struct_size,
            params.bboxes_num
        );
        
        // --- NMS 后处理 ---
        frame->yolo_list = yoloArmorNMS_V5(
            armor_output_host_buffer, params.bboxes_num, params.class_num,
            params.conf_thresh, params.nms_thresh, frame->width, frame->height,
            params.infer_width, params.infer_height
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
    
    rm::message("Cleanup complete. Exiting.", rm::MSG_NOTE);
    return 0;
}