    #include <iostream>
    #include <string>
    #include <vector>
    #include <stdexcept>
    #include <filesystem> // C++17, 用于文件路径操作
    #include <fstream>    // **[修复 1]** 必须包含，用于 std::ifstream

    // OpenCV 头文件
    #include <opencv2/opencv.hpp>

    // YOLOv5-TensorRT 库的头文件
    #include <yolov5_builder.hpp>
    #include <yolov5_detector.hpp>

    // ====================================================================================
    // 1. 参数结构体 (AppParams)
    // ====================================================================================
    struct AppParams {
        // --- 模型相关参数 ---
        std::string onnx_path;          // ONNX模型文件的路径 (必需)
        int infer_width       = 416;    // 模型推理所需的输入图像宽度 (注意：此处未被Detector直接使用，但保留作为配置)
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
        // **[修复 1]** std::ifstream f(name.c_str());
        std::ifstream f(name);
        return f.good();
    }

    // ====================================================================================
    // 3. 绘制边界框函数
    // ====================================================================================
    void draw_bboxes(cv::Mat& image, const std::vector<yolov5::Detection>& detections) {
        for (const auto& det : detections) {
            // **[修复 2a]** 使用 classId() 方法获取类别ID
            int class_id = det.classId();
            // **[修复 2b]** 使用 boundingBox() 方法获取边界框
            const cv::Rect& box = det.boundingBox();
            // **[修复 2c]** 使用 score() 方法获取置信度
            double score = det.score();

            // 安全检查，防止类别ID越界
            if (class_id < 0 || class_id >= CLASS_NAMES.size()) continue;

            // 绘制矩形框
            cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2);

            // 准备标签文本，包含类别名和置信度
            std::string label = CLASS_NAMES[class_id] + ": " + cv::format("%.2f", score);

            // 计算文本尺寸以便绘制背景
            int baseline;
            cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            
            // 绘制文本背景框
            cv::rectangle(image,
                        cv::Point(box.x, box.y - label_size.height - baseline),
                        cv::Point(box.x + label_size.width, box.y),
                        cv::Scalar(0, 255, 0),
                        cv::FILLED);

            // 放置文本
            cv::putText(image, label,
                        cv::Point(box.x, box.y - baseline),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }
    }

    // ====================================================================================
    // 4. 主函数 (main)
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
std::cout << "Debug: onnx_path from params: [" << params.onnx_path << "]" << std::endl;

if (params.onnx_path.empty()) {
    std::cerr << "错误: 模型路径为空，程序终止。" << std::endl;
    return -1;
}

// 步骤 1: 创建 path 对象
std::filesystem::path onnx_filepath(params.onnx_path);
std::cout << "Debug: path object created successfully." << std::endl;
std::cout << "ONNX filepath: " << onnx_filepath.string() << std::endl;

// 步骤 2: 替换扩展名
std::filesystem::path engine_filepath_path = onnx_filepath.replace_extension(".engine");
std::cout << "Debug: replace_extension successful." << std::endl;

// 步骤 3: 转换为字符串
std::string engine_filepath = engine_filepath_path.string();
std::cout << "Debug: .string() conversion successful." << std::endl;

std::cout << "TensorRT engine filepath: " << engine_filepath << std::endl;

        std::cout << "TensorRT engine filepath: " << engine_filepath << std::endl;
        // 如果 .engine 文件不存在，则从 ONNX 文件构建它
        if (!file_exists(engine_filepath)) {
            std::cout << "TensorRT engine file not found at: " << engine_filepath << std::endl;
            std::cout << "Building engine from ONNX file: " << params.onnx_path << "..." << std::endl;
            
            try {
                yolov5::Builder builder;
                // 确保 Builder 初始化成功
                if (builder.init() != yolov5::RESULT_SUCCESS) {
                    std::cerr << "Error: yolov5::Builder init failed." << std::endl;
                    return -1;
                }

                // **[修复 3]** Builder 的构建方法是 buildEngine
                yolov5::Result build_res = builder.buildEngine(params.onnx_path, engine_filepath);
                // 您可以在这里设置其他构建参数，例如 FP16 模式:
                // yolov5::Result build_res = builder.buildEngine(params.onnx_path, engine_filepath, yolov5::PRECISION_FP16);

                if (build_res != yolov5::RESULT_SUCCESS) {
                    std::cerr << "Error building TensorRT engine. Result code: " << build_res << std::endl;
                    return -1;
                }

                std::cout << "Engine built successfully and saved to " << engine_filepath << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Error building TensorRT engine (exception): " << e.what() << std::endl;
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
            
            // **[修复 4a]** init() 不接受阈值参数，它只接受 flags
            if (detector->init() != yolov5::RESULT_SUCCESS) {
                std::cerr << "Error: yolov5::Detector init failed." << std::endl;
                return -1;
            }
            
            // **[修复 4b]** 通过 setScoreThreshold 和 setNmsThreshold 设置阈值
            float conf_thresh = 0.4;
            float nms_thresh = 0.5;
            detector->setScoreThreshold(conf_thresh);
            detector->setNmsThreshold(nms_thresh);

            // 加载类别名称
            yolov5::Classes classes;
            classes.load(CLASS_NAMES);
            detector->setClasses(classes);
            
            // 加载引擎
            if (detector->loadEngine(engine_filepath) != yolov5::RESULT_SUCCESS) {
                std::cerr << "Error: Failed to load TensorRT engine." << std::endl;
                return -1;
            }

            std::cout << "Detector initialized successfully." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error initializing detector (exception): " << e.what() << std::endl;
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
        
    { //创建一个局部作用域来管理 detector 的生命周期
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
            // 使用 YOLOv5-TensorRT 库进行检测
            if (detector->detect(frame, &detections) != yolov5::RESULT_SUCCESS) {
                std::cerr << "Warning: Detection failed for a frame." << std::endl;
                // 继续下一帧
            }
            
            // --- 绘制结果 ---
            draw_bboxes(frame, detections);

            // --- 显示画面 ---
            cv::imshow(window_name, frame);

            // 按 ESC 键退出
            if (cv::waitKey(1) == 27) {
                break;
            }
        }

    }

        // 6. 释放资源
        std::cout << "Releasing resources..." << std::endl;
        cap.release();
        cv::destroyAllWindows();
        std::cout << "Done." << std::endl;

        return 0;
    }