#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <filesystem> // C++17, 用于文件路径操作

// OpenCV 头文件
#include <opencv2/opencv.hpp>

// YOLOv5-TensorRT 库的头文件
#include <yolov5_builder.hpp>
#include <yolov5_detector.hpp>

// ====================================================================================
// 1. 参数结构体 (AppParams)
//    - 沿用您参考代码中的结构，集中管理所有配置参数。
// ====================================================================================
struct AppParams {
    // --- 模型相关参数 ---
    std::string onnx_path;          // ONNX模型文件的路径 (必需)
    int infer_width       = 416;    // 模型推理所需的输入图像宽度
    int infer_height      = 416;    // 模型推理所需的输入图像高度
    // 注意: YOLOv5-TensorRT库会自动处理置信度和NMS阈值，
    // 我们可以在 Detector 初始化时设置它们。
};

// 类别名称 - 请确保顺序与模型训练时完全一致
const std::vector<std::string> CLASS_NAMES = {
    "B1", "B2", "B3", "B4", "B5", "BHero", 
    "R1", "R2", "R3", "R4", "R5", "RHero", 
    "RQS", "BQS"
};

// ====================================================================================
// 2. 帮助与命令行解析函数
// ====================================================================================

void print_usage(const char* prog_name) {
    std::cout << "\n用法: " << prog_name << " -m <path_to_model.onnx> [可选参数]\n\n"
              << "必需参数:\n"
              << "  -m, --model <path>      ONNX模型文件的路径。\n\n"
              << "可选模型参数:\n"
              << "  --width <int>           模型推理宽度 (默认: 416)\n"
              << "  --height <int>          模型推理高度 (默认: 416)\n\n"
              << "其他:\n"
              << "  -h, --help              显示此帮助信息。\n" << std::endl;
}

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
        } else {
            std::cerr << "错误: 未知或不完整的参数: " << arg << std::endl;
            print_usage(argv[0]);
            exit(1);
        }
    }
}

// 检查文件是否存在的辅助函数
bool file_exists(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

// ====================================================================================
// 3. 绘制边界框函数
//    - 功能: 在图像上绘制检测到的边界框和标签。
//    - 已适配为使用 yolov5::Detection 结构体。
// ====================================================================================
void draw_bboxes(cv::Mat& image, const std::vector<yolov5::Detection>& detections) {
    for (const auto& det : detections) {
        // 安全检查，防止类别ID越界
        if (det.classId >= CLASS_NAMES.size()) continue;

        // 绘制矩形框
        cv::rectangle(image, det.box, cv::Scalar(0, 255, 0), 2);

        // 准备标签文本，包含类别名和置信度
        std::string label = CLASS_NAMES[det.classId] + ": " + cv::format("%.2f", det.confidence);

        // 计算文本尺寸以便绘制背景
        int baseline;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        
        // 绘制文本背景框
        cv::rectangle(image,
                      cv::Point(det.box.x, det.box.y - label_size.height - baseline),
                      cv::Point(det.box.x + label_size.width, det.box.y),
                      cv::Scalar(0, 255, 0),
                      cv::FILLED);

        // 放置文本
        cv::putText(image, label,
                    cv::Point(det.box.x, det.box.y - baseline),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

// ====================================================================================
// 4. 主函数 (main)
//    - 整个应用程序的入口点和主流程控制器。
// ====================================================================================
int main(int argc, char* argv[]) {
    // 1. 初始化与参数解析
    AppParams params;
    parse_arguments(argc, argv, params);
    
    if (params.onnx_path.empty()) {
        std::cerr << "错误: 必须通过 -m 或 --model 提供模型路径。" << std::endl;
        print_usage(argv[0]);
        return -1;
    }

    // 2. 准备 TensorRT 引擎文件
    // 自动生成 .engine 文件路径
    std::filesystem::path onnx_filepath(params.onnx_path);
    std::string engine_filepath = onnx_filepath.replace_extension(".engine").string();

    // 如果 .engine 文件不存在，则从 ONNX 文件构建它
    if (!file_exists(engine_filepath)) {
        std::cout << "TensorRT engine file not found at: " << engine_filepath << std::endl;
        std::cout << "Building engine from ONNX file: " << params.onnx_path << "..." << std::endl;
        
        try {
            yolov5::Builder builder;
            builder.init();
            // 您可以在这里设置其他构建参数，例如 FP16 模式
            // builder.build(params.onnx_path, engine_filepath, yolov5::Precision::FP16);
            builder.build(params.onnx_path, engine_filepath);
            std::cout << "Engine built successfully and saved to " << engine_filepath << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error building TensorRT engine: " << e.what() << std::endl;
            return -1;
        }
    } else {
        std::cout << "Found existing TensorRT engine file: " << engine_filepath << std::endl;
    }

    // 3. 初始化 YOLOv5 检测器
    std::unique_ptr<yolov5::Detector> detector;
    try {
        std::cout << "Initializing YOLOv5 detector..." << std::endl;
        detector = std::make_unique<yolov5::Detector>();
        
        // 设置检测器参数（可以从命令行读取，此处使用默认值）
        float conf_thresh = 0.4;
        float nms_thresh = 0.5;
        detector->init(conf_thresh, nms_thresh);
        
        // 加载引擎
        detector->loadEngine(engine_filepath);
        std::cout << "Detector initialized successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing detector: " << e.what() << std::endl;
        return -1;
    }

    // 4. 初始化摄像头
    std::cout << "Opening camera..." << std::endl;
    cv::VideoCapture cap(0); // 0 代表默认摄像头
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }
    std::cout << "Camera opened successfully." << std::endl;

    // 5. 主循环 - 实时检测
    const std::string window_name = "YOLOv5-TensorRT Real-time Detection";
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

    cv::Mat frame;
    std::vector<yolov5::Detection> detections;

    std::cout << "Starting real-time detection... Press 'ESC' to exit." << std::endl;
    while (true) {
        // 从摄像头捕获一帧
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Warning: Captured empty frame." << std::endl;
            break;
        }

        // --- 核心推理 ---
        // 使用 YOLOv5-TensorRT 库进行检测，所有复杂的预处理、推理和后处理都被封装在这一行代码中！
        detector->detect(frame, &detections);
        
        // --- 绘制结果 ---
        draw_bboxes(frame, detections);

        // --- 显示画面 ---
        cv::imshow(window_name, frame);

        // 按 ESC 键退出
        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    // 6. 释放资源
    std::cout << "Releasing resources..." << std::endl;
    cap.release();
    cv::destroyAllWindows();
    std::cout << "Done." << std::endl;

    return 0;
}