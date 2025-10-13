/**
 * @file main.cpp
 * @brief 使用 YOLOv5-TensorRT 和海康相机进行实时目标检测的示例程序。
 *
 * 该程序通过命令行参数接收ONNX模型路径和相关配置，
 * 动态构建或加载TensorRT引擎，然后初始化海康工业相机进行实时视频流捕获。
 * 捕获的每一帧都会通过YOLOv5模型进行推理，检测结果（边界框和类别）
 * 将被绘制在图像上并实时显示。程序还提供了实时帧率监控功能。
 *
 * 主要依赖库:
 * - OpenCV: 用于图像处理和显示。
 * - YOLOv5-TensorRT: 用于模型推理加速。
 * - DpRM (RM 库): 用于海康相机的控制。
 * - C++ Standard Library: 用于文件操作、多线程、时间等。
 */

// ====================================================================================
// 依赖头文件
// ====================================================================================

// C++ 标准库
#include <iostream>      // 用于标准输入输出 (std::cout, std::cerr)
#include <string>        // 用于字符串处理 (std::string)
#include <vector>        // 用于动态数组 (std::vector)
#include <stdexcept>     // 用于标准异常处理
#include <filesystem>    // 用于文件路径操作
#include <fstream>       // 用于文件流操作 (std::ifstream)
#include <memory>        // 用于智能指针 (std::unique_ptr, std::shared_ptr)
#include <thread>        // 用于线程相关操作 (std::this_thread)
#include <chrono>        // 用于高精度时间测量 (std::chrono)
#include <iomanip>       // 用于输出格式化 (std::setprecision)

// 第三方库头文件
#include <opencv2/opencv.hpp>        // OpenCV核心库

#include <yolov5_builder.hpp>        // YOLOv5-TensorRT 引擎构建器
#include <yolov5_detector.hpp>       // YOLOv5-TensorRT 检测器

// 假设的RM库头文件 (用于海康相机)
// 确保这些头文件的路径已添加到编译器的包含目录中
#include <dprm/dprm.h>

// ====================================================================================
// 1. 全局配置与定义
// ====================================================================================

/**
 * @struct AppParams
 * @brief 存储应用程序的所有可配置参数。
 *
 * 该结构体整合了模型推理、相机设置以及其他行为相关的参数，
 * 便于通过命令行进行统一管理。
 */
struct AppParams {
    // --- 模型相关参数 ---
    std::string onnx_path;          ///< ONNX模型文件的路径
    int infer_width       = 416;    ///< 模型推理的输入宽度
    int infer_height      = 416;    ///< 模型推理的输入高度
    double conf_thresh    = 0.4;    ///< 目标检测的置信度阈值
    double nms_thresh     = 0.5;    ///< 非极大值抑制 (NMS) 的IOU阈值

    // --- 相机相关参数 ---
    double exposure       = 2500.0; ///< 相机曝光时间 (单位: 微秒)
    double gain           = 12.0;   ///< 相机增益
    double gamma          = 200.0;  ///< 相机Gamma值

    // --- 功能开关 ---
    bool show_fps         = false;  ///< 是否在控制台显示实时帧率
};

/**
 * @brief 模型类别名称列表。
 * @note 列表顺序必须与模型训练时定义的类别顺序完全一致。
 */
const std::vector<std::string> CLASS_NAMES = {
    "B1", "B2", "B3", "B4", "B5", "BHero",
    "R1", "R2", "R3", "R4", "R5", "RHero",
    "RQS", "BQS"
};

// ====================================================================================
// 2. 辅助函数
// ====================================================================================

/**
 * @brief 打印程序的用法说明到控制台。
 * @param prog_name 程序的执行文件名 (通常是 argv[0])。
 */
void print_usage(const char* prog_name) {
    AppParams defaults; // 创建一个默认参数实例以便在帮助信息中显示默认值
    std::cout << "\n用法: " << prog_name << " -m <path_to_model.onnx> [可选参数]\n\n"
              << "必需参数:\n"
              << "  -m, --model <path>      ONNX模型文件的路径。\n\n"
              << "可选模型参数:\n"
              << "  --width <int>           模型推理宽度 (默认: " << defaults.infer_width << ")\n"
              << "  --height <int>          模型推理高度 (默认: " << defaults.infer_height << ")\n"
              << "  --conf <float>          置信度阈值 (默认: " << defaults.conf_thresh << ")\n"
              << "  --nms <float>           NMS阈值 (默认: " << defaults.nms_thresh << ")\n\n"
              << "可选相机参数:\n"
              << "  --exposure <float>      相机曝光时间 (默认: " << defaults.exposure << ")\n"
              << "  --gain <float>          相机增益 (默认: " << defaults.gain << ")\n"
              << "  --gamma <float>         相机Gamma值 (默认: " << defaults.gamma << ")\n\n"
              << "其他:\n"
              << "  --show-fps              在标准输出中实时显示帧率。\n"
              << "  -h, --help              显示此帮助信息。\n" << std::endl;
}

/**
 * @brief 解析命令行参数并填充到 AppParams 结构体中。
 * @param argc 参数数量 (来自 main 函数)。
 * @param argv 参数值数组 (来自 main 函数)。
 * @param params [out] 用于接收解析结果的 AppParams 结构体引用。
 */
void parse_arguments(int argc, char* argv[], AppParams& params) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            params.onnx_path = argv[++i];
        } else if (arg == "--width" && i + 1 < argc) {
            params.infer_width = std::stoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            params.infer_height = std::stoi(argv[++i]);
        } else if (arg == "--exposure" && i + 1 < argc) {
            params.exposure = std::stod(argv[++i]);
        } else if (arg == "--gain" && i + 1 < argc) {
            params.gain = std::stod(argv[++i]);
        } else if (arg == "--gamma" && i + 1 < argc) {
            params.gamma = std::stod(argv[++i]);
        } else if (arg == "--conf" && i + 1 < argc) {
            params.conf_thresh = std::stod(argv[++i]);
        } else if (arg == "--nms" && i + 1 < argc) {
            params.nms_thresh = std::stod(argv[++i]);
        }
        else if (arg == "--show-fps") {
            params.show_fps = true;
        }
        else {
            std::cerr << "错误: 未知或不完整的参数: " << arg << std::endl;
            print_usage(argv[0]);
            exit(1);
        }
    }
}

/**
 * @brief 检查指定路径的文件是否存在。
 * @param name 文件的完整路径。
 * @return 如果文件存在且可读，返回 true，否则返回 false。
 */
bool file_exists(const std::string& name) {
    std::ifstream f(name);
    return f.good();
}


/**
 * @brief 在给定的图像上绘制检测到的边界框。
 * @param image [in, out] 要在其上绘制的图像 (cv::Mat)。
 * @param detections 检测结果的向量。
 */
void draw_bboxes(cv::Mat& image, const std::vector<yolov5::Detection>& detections) {
    for (const auto& det : detections) {
        // 获取检测结果的基本信息
        int class_id = det.classId();
        const cv::Rect& box = det.boundingBox();
        double score = det.score();

        // 安全检查，防止类别ID越界
        if (class_id < 0 || class_id >= CLASS_NAMES.size()) continue;

        // 绘制边界框
        cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2);

        // 准备标签文本，格式为 "类别名: 置信度"
        std::string label = CLASS_NAMES[class_id] + ": " + cv::format("%.2f", score);

        // 计算标签文本的尺寸以便绘制背景
        int baseline;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        
        // 绘制标签背景框
        cv::rectangle(image,
                    cv::Point(box.x, box.y - label_size.height - baseline),
                    cv::Point(box.x + label_size.width, box.y),
                    cv::Scalar(0, 255, 0),
                    cv::FILLED);

        // 绘制标签文本
        cv::putText(image, label,
                    cv::Point(box.x, box.y - baseline),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

// ====================================================================================
// 3. 主函数 (main)
// ====================================================================================
int main(int argc, char* argv[]) {
    // -----------------------------------------------------------------
    // 步骤 1: 初始化与参数解析
    // -----------------------------------------------------------------
    AppParams params;
    parse_arguments(argc, argv, params);
    
    // 验证必需参数：模型路径
    if (params.onnx_path.empty()) {
        std::cerr << "错误: 必须通过 -m 或 --model 提供模型路径。" << std::endl;
        print_usage(argv[0]);
        return -1;
    }

    // 验证模型文件是否存在
    if (!file_exists(params.onnx_path)) {
        std::cerr << "错误: 提供的ONNX文件不存在: " << params.onnx_path << std::endl;
        return -1;
    }

    // -----------------------------------------------------------------
    // 步骤 2: 准备 TensorRT 引擎文件
    // -----------------------------------------------------------------
    // 自动生成引擎文件的路径 (例如, a.onnx -> a.engine)
    std::string engine_filepath;
    std::string onnx_path_str = params.onnx_path;
    size_t last_dot_pos = onnx_path_str.find_last_of(".");
    if (last_dot_pos != std::string::npos) {
        engine_filepath = onnx_path_str.substr(0, last_dot_pos) + ".engine";
    } else {
        engine_filepath = onnx_path_str + ".engine";
    }
    
    std::cout << "ONNX文件路径: " << onnx_path_str << std::endl;
    std::cout << "TensorRT引擎文件路径: " << engine_filepath << std::endl;
    
    // 如果引擎文件不存在，则从ONNX文件构建
    if (!file_exists(engine_filepath)) {
        std::cout << "未找到TensorRT引擎文件，正在从ONNX文件构建..." << std::endl;
        try {
            yolov5::Builder builder;
            if (builder.init() != yolov5::RESULT_SUCCESS) {
                std::cerr << "错误: yolov5::Builder 初始化失败。" << std::endl;
                return -1;
            }
            yolov5::Result build_res = builder.buildEngine(params.onnx_path, engine_filepath);
            if (build_res != yolov5::RESULT_SUCCESS) {
                std::cerr << "错误: 构建TensorRT引擎失败。结果代码: " << build_res << std::endl;
                return -1;
            }
            std::cout << "引擎构建成功并保存至 " << engine_filepath << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "错误: 构建TensorRT引擎时发生异常: " << e.what() << std::endl;
            return -1;
        }
    } else {
        std::cout << "找到已存在的TensorRT引擎文件: " << engine_filepath << std::endl;
    }

    // -----------------------------------------------------------------
    // 步骤 3: 初始化 YOLOv5 检测器
    // -----------------------------------------------------------------
    std::unique_ptr<yolov5::Detector> detector;
    try {
        std::cout << "正在初始化YOLOv5检测器..." << std::endl;
        detector = std::make_unique<yolov5::Detector>();
        
        // 初始化检测器
        if (detector->init() != yolov5::RESULT_SUCCESS) {
            std::cerr << "错误: yolov5::Detector 初始化失败。" << std::endl;
            return -1;
        }
        
        // 设置推理参数
        detector->setScoreThreshold(params.conf_thresh);
        detector->setNmsThreshold(params.nms_thresh);
        std::cout << "设置置信度阈值 (conf): " << params.conf_thresh << std::endl;
        std::cout << "设置NMS阈值 (nms): " << params.nms_thresh << std::endl;

        // 加载类别名称
        yolov5::Classes classes;
        classes.load(CLASS_NAMES);
        detector->setClasses(classes);
        
        // 加载 TensorRT 引擎
        if (detector->loadEngine(engine_filepath) != yolov5::RESULT_SUCCESS) {
            std::cerr << "错误: 加载TensorRT引擎失败。" << std::endl;
            return -1;
        }
        std::cout << "检测器初始化成功。" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "错误: 初始化检测器时发生异常: " << e.what() << std::endl;
        return -1;
    }

    // -----------------------------------------------------------------
    // 步骤 4: 初始化 HIK 相机
    // -----------------------------------------------------------------
    std::cout << "正在初始化HIK相机..." << std::endl;
    rm::Camera* camera = new rm::Camera(); // 注意: 手动管理内存
    int camera_num = -1;
    // 检查是否能找到海康相机
    if (!rm::getHikCameraNum(camera_num) || camera_num < 1) {
        std::cerr << "错误: 获取相机数量失败或未找到相机。" << std::endl;
        delete camera;
        return -1;
    }

    // 打开第一个找到的相机并设置参数
    if (!rm::openHik(camera, 1, nullptr, nullptr, nullptr, params.exposure, params.gain, params.gamma)) {
        std::cerr << "错误: 打开相机1失败。" << std::endl;
        delete camera;
        return -1;
    }
    std::cout << "相机打开成功。" << std::endl;

    // -----------------------------------------------------------------
    // 步骤 5: 主循环 - 实时捕获与检测
    // -----------------------------------------------------------------
    const std::string window_name = "YOLOv5-TensorRT Real-time Detection";
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

    cv::Mat frame_mat;
    std::vector<yolov5::Detection> detections;
    auto frame_wait_tp = getTime(); // 假设 getTime() 是一个返回当前时间的函数

    // 用于计算帧率的变量
    int camera_frame_count = 0; // 记录相机成功捕获的帧数
    int yolo_frame_count = 0;   // 记录YOLO成功推理的帧数
    double yolo_total_time_ms = 0.0; // 记录YOLO总推理耗时（毫秒）
    auto fps_start_time = std::chrono::high_resolution_clock::now(); // FPS计算的起始时间点

    std::cout << "开始实时检测... 按 'ESC' 键退出。" << std::endl;
    while (true) {
        // 从HIK相机缓冲区弹出一帧图像
        std::shared_ptr<rm::Frame> frame_ptr = camera->buffer->pop();
        
        // 如果帧无效，进行超时检查
        if (frame_ptr == nullptr || !frame_ptr->image || frame_ptr->image->empty()) {
            if (getDoubleOfS(frame_wait_tp, getTime()) > 2.0) { // 假设 getDoubleOfS 计算时间差（秒）
                std::cerr << "错误: 图像捕获超时!" << std::endl;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1)); // 短暂休眠，避免CPU空转
            continue;
        }
        // 重置超时计时器，并增加相机帧计数
        frame_wait_tp = getTime();
        camera_frame_count++;

        frame_mat = *(frame_ptr->image);

        // --- 核心推理与计时 ---
        auto yolo_start_time = std::chrono::high_resolution_clock::now();
        bool detection_success = (detector->detect(frame_mat, &detections) == yolov5::RESULT_SUCCESS);
        auto yolo_end_time = std::chrono::high_resolution_clock::now();
        
        if (detection_success) {
            // 累加成功的推理耗时和次数
            std::chrono::duration<double, std::milli> yolo_duration = yolo_end_time - yolo_start_time;
            yolo_total_time_ms += yolo_duration.count();
            yolo_frame_count++;
        } else {
            std::cerr << "警告: 对当前帧进行检测失败。" << std::endl;
        }
        
        // --- 帧率计算与输出 ---
        if (params.show_fps) {
            auto current_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = current_time - fps_start_time;

            // 每隔 1 秒更新一次控制台输出
            if (elapsed.count() >= 1.0) {
                // 计算相机和YOLO的平均帧率
                double camera_fps = camera_frame_count / elapsed.count();
                double yolo_fps = (yolo_frame_count > 0) ? (yolo_frame_count * 1000.0 / yolo_total_time_ms) : 0.0;

                // 使用 '\r' 和 std::flush 在同一行覆盖刷新信息，避免刷屏
                std::cout << "\r相机 FPS: " << std::fixed << std::setprecision(2) << camera_fps
                          << " | YOLO 推理 FPS: " << std::fixed << std::setprecision(2) << yolo_fps
                          << "      " << std::flush;

                // 重置计数器和计时器，为下一个周期做准备
                camera_frame_count = 0;
                yolo_frame_count = 0;
                yolo_total_time_ms = 0.0;
                fps_start_time = current_time;
            }
        }

        // --- 结果可视化 ---
        draw_bboxes(frame_mat, detections);

        // --- 显示图像 ---
        cv::imshow(window_name, frame_mat);

        // 检测按键，如果按下 'ESC' (ASCII 27)，则退出循环
        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    // -----------------------------------------------------------------
    // 步骤 6: 释放资源
    // -----------------------------------------------------------------
    if(params.show_fps) std::cout << std::endl; // 帧率显示后换行，使后续输出整洁
    std::cout << "正在释放资源..." << std::endl;
    delete camera; // 释放相机对象
    cv::destroyAllWindows(); // 关闭所有OpenCV窗口
    std::cout << "完成。" << std::endl;

    return 0;
}