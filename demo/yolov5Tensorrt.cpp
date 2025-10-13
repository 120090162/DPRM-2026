#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <memory>    // 用于 std::unique_ptr 和 std::shared_ptr
#include <thread>    // 用于 std::this_thread
#include <chrono>    // 用于 std::chrono
#include <iomanip>   // [新增] 用于 std::setprecision, 精确控制输出浮点数

// OpenCV 头文件
#include <opencv2/opencv.hpp>

// YOLOv5-TensorRT 库的头文件
#include <yolov5_builder.hpp>
#include <yolov5_detector.hpp>

// 示例代码中的 RM 库头文件 (用于HIK相机)
// *** 确保这些头文件的路径在您的编译器包含路径中 ***
#include <dprm/dprm.h>

// ====================================================================================
// 1. 参数结构体 (AppParams) - 已修改
// ====================================================================================
struct AppParams {
    // --- 模型相关参数 ---
    std::string onnx_path;
    int infer_width       = 416;
    int infer_height      = 416;
    double conf_thresh    = 0.4;
    double nms_thresh     = 0.5;

    // --- 相机相关参数 ---
    double exposure       = 2500.0;
    double gain           = 12.0;
    double gamma          = 200.0;

    // --- [新增] 显示帧率的标志 ---
    bool show_fps         = false;
};

// 类别名称
const std::vector<std::string> CLASS_NAMES = {
    "B1", "B2", "B3", "B4", "B5", "BHero",
    "R1", "R2", "R3", "R4", "R5", "RHero",
    "RQS", "BQS"
};

// ====================================================================================
// 2. 帮助与命令行解析函数 - 已修改
// ====================================================================================
void print_usage(const char* prog_name) {
    AppParams defaults; // 用于显示默认值
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
              << "  --show-fps              [新增] 在标准输出中实时显示帧率。\n" // [新增]
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
        // [新增] 添加对 --show-fps 参数的解析
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

bool file_exists(const std::string& name) {
    std::ifstream f(name);
    return f.good();
}

// ====================================================================================
// 3. 绘制边界框函数 (与您的代码相同)
// ====================================================================================
void draw_bboxes(cv::Mat& image, const std::vector<yolov5::Detection>& detections) {
    for (const auto& det : detections) {
        int class_id = det.classId();
        const cv::Rect& box = det.boundingBox();
        double score = det.score();

        if (class_id < 0 || class_id >= CLASS_NAMES.size()) continue;

        cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2);

        std::string label = CLASS_NAMES[class_id] + ": " + cv::format("%.2f", score);

        int baseline;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        
        cv::rectangle(image,
                    cv::Point(box.x, box.y - label_size.height - baseline),
                    cv::Point(box.x + label_size.width, box.y),
                    cv::Scalar(0, 255, 0),
                    cv::FILLED);

        cv::putText(image, label,
                    cv::Point(box.x, box.y - baseline),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

// ====================================================================================
// 4. 主函数 (main) - 已修改
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

    if (!file_exists(params.onnx_path)) {
        std::cerr << "错误: 提供的ONNX文件不存在: " << params.onnx_path << std::endl;
        return -1;
    }

    // 2. 准备 TensorRT 引擎文件
    std::string engine_filepath;
    std::string onnx_path_str = params.onnx_path;
    size_t last_dot_pos = onnx_path_str.find_last_of(".");
    if (last_dot_pos != std::string::npos) {
        engine_filepath = onnx_path_str.substr(0, last_dot_pos) + ".engine";
    } else {
        engine_filepath = onnx_path_str + ".engine";
    }
    
    std::cout << "ONNX filepath: " << onnx_path_str << std::endl;
    std::cout << "TensorRT engine filepath: " << engine_filepath << std::endl;
    
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

    // 3. 初始化 YOLOv5 检测器
    std::unique_ptr<yolov5::Detector> detector;
    try {
        std::cout << "正在初始化YOLOv5检测器..." << std::endl;
        detector = std::make_unique<yolov5::Detector>();
        
        if (detector->init() != yolov5::RESULT_SUCCESS) {
            std::cerr << "错误: yolov5::Detector 初始化失败。" << std::endl;
            return -1;
        }
        
        detector->setScoreThreshold(params.conf_thresh);
        detector->setNmsThreshold(params.nms_thresh);
        std::cout << "设置置信度阈值 (conf): " << params.conf_thresh << std::endl;
        std::cout << "设置NMS阈值 (nms): " << params.nms_thresh << std::endl;

        yolov5::Classes classes;
        classes.load(CLASS_NAMES);
        detector->setClasses(classes);
        
        if (detector->loadEngine(engine_filepath) != yolov5::RESULT_SUCCESS) {
            std::cerr << "错误: 加载TensorRT引擎失败。" << std::endl;
            return -1;
        }
        std::cout << "检测器初始化成功。" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "错误: 初始化检测器时发生异常: " << e.what() << std::endl;
        return -1;
    }

    // 4. 初始化 HIK 相机
    std::cout << "正在初始化HIK相机..." << std::endl;
    rm::Camera* camera = new rm::Camera();
    int camera_num = -1;
    if (!rm::getHikCameraNum(camera_num) || camera_num < 1) {
        std::cerr << "错误: 获取相机数量失败或未找到相机。" << std::endl;
        delete camera;
        return -1;
    }

    if (!rm::openHik(camera, 1, nullptr, nullptr, nullptr, params.exposure, params.gain, params.gamma)) {
        std::cerr << "错误: 打开相机1失败。" << std::endl;
        delete camera;
        return -1;
    }
    std::cout << "相机打开成功。" << std::endl;

    // 5. 主循环 - 实时检测
    const std::string window_name = "YOLOv5-TensorRT Real-time Detection";
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

    cv::Mat frame_mat;
    std::vector<yolov5::Detection> detections;
    auto frame_wait_tp = getTime();

    // [新增] 用于计算帧率的变量
    int camera_frame_count = 0;
    int yolo_frame_count = 0;
    double yolo_total_time_ms = 0.0;
    auto fps_start_time = std::chrono::high_resolution_clock::now();


    std::cout << "开始实时检测... 按 'ESC' 键退出。" << std::endl;
    while (true) {
        // 从HIK相机缓冲区捕获一帧
        std::shared_ptr<rm::Frame> frame_ptr = camera->buffer->pop();
        
        if (frame_ptr == nullptr || !frame_ptr->image || frame_ptr->image->empty()) {
            if (getDoubleOfS(frame_wait_tp, getTime()) > 2.0) {
                std::cerr << "错误: 图像捕获超时!" << std::endl;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        frame_wait_tp = getTime();
        camera_frame_count++; // [新增] 摄像头帧计数

        frame_mat = *(frame_ptr->image);

        // --- 核心推理 ---
        auto yolo_start_time = std::chrono::high_resolution_clock::now(); // [新增]
        bool detection_success = (detector->detect(frame_mat, &detections) == yolov5::RESULT_SUCCESS);
        auto yolo_end_time = std::chrono::high_resolution_clock::now(); // [新增]
        
        if (detection_success) {
            // [新增] 累加成功的推理时间和计数
            std::chrono::duration<double, std::milli> yolo_duration = yolo_end_time - yolo_start_time;
            yolo_total_time_ms += yolo_duration.count();
            yolo_frame_count++;
        } else {
            std::cerr << "警告: 对当前帧进行检测失败。" << std::endl;
        }
        
        // --- [新增] 帧率计算与输出逻辑 ---
        if (params.show_fps) {
            auto current_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = current_time - fps_start_time;

            // 每隔1秒更新一次输出
            if (elapsed.count() >= 1.0) {
                double camera_fps = camera_frame_count / elapsed.count();
                // 如果yolo成功推理次数大于0，则计算平均帧率，否则为0
                double yolo_fps = (yolo_frame_count > 0) ? (yolo_frame_count * 1000.0 / yolo_total_time_ms) : 0.0;

                // 使用 \r 和 std::flush 在同一行更新信息
                std::cout << "\rCamera FPS: " << std::fixed << std::setprecision(2) << camera_fps
                          << " | YOLO FPS: " << std::fixed << std::setprecision(2) << yolo_fps
                          << "      " << std::flush;

                // 重置计数器和计时器
                camera_frame_count = 0;
                yolo_frame_count = 0;
                yolo_total_time_ms = 0.0;
                fps_start_time = current_time;
            }
        }

        // --- 绘制结果 ---
        draw_bboxes(frame_mat, detections);

        // --- 显示画面 ---
        cv::imshow(window_name, frame_mat);

        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    // 6. 释放资源
    if(params.show_fps) std::cout << std::endl; // [新增] 换行，避免最后一行帧率信息被覆盖
    std::cout << "正在释放资源..." << std::endl;
    delete camera;
    cv::destroyAllWindows();
    std::cout << "完成。" << std::endl;

    return 0;
}