// 总结：TensorRT API的“代沟”问题
// 问题的根源在于，TensorRT从早期版本（如v7, v8）到现代版本（v9, v10）进行了一次重要的API设计哲学变更。您的C++代码（特别是rm库）遵循的是旧版逻辑，而在新版环境中运行时，虽然编译能通过，但无法正确执行。
// 错误 1: 获取张量形状 (Shape) 的方式
// 旧版API (get_binding_shape):
// 工作方式: 通过一个整数索引（index）来引用输入/输出张量，例如 engine->getBindingDimensions(0)。
// 缺点: 这种方式是“脆弱”的。如果模型的输入输出顺序发生变化，或者增加了新的输入/输出，代码中的索引就可能指向错误的张量，导致难以察觉的bug。
// 新版API (get_tensor_shape):
// 工作方式: 通过一个字符串名称（name）来引用张量，例如 engine->getTensorShape("images")。
// 优点: 这种方式非常稳健和自文档化。无论模型内部如何排序，只要张量名称不变，代码就能准确地找到它，大大提高了代码的可读性和可靠性。
// 对您C++代码的影响:
// 您的C++代码很可能仍在使用基于索引的旧方法来获取输入/输出缓冲区的大小。在新版API中，虽然旧函数可能为了兼容性依然存在，但推荐使用新方法。我们Python脚本的第一个错误正是因为get_binding_shape函数在新API中已被移除或重命名。
// 错误 2: 执行推理 (Execution) 的工作流程
// 这是导致您“输出为零”的最核心原因。
// 旧版API (enqueueV2, execute_async + bindings):
// 工作方式: 在每次执行推理时，通过一个 bindings 数组（一个存放GPU内存地址指针的数组）来告诉引擎“这次推理请用这些内存地址作为输入和输出”。
// 流程: 准备数据 -> 准备bindings数组 -> 调用enqueue(bindings)。
// 新版API (set_tensor_address + enqueueV3 / execute_async_v3):
// 工作方式: 采用“一次性设置，多次运行”的模式。
// 配置阶段: 在创建执行上下文(IExecutionContext)后，就提前通过 context->setInputTensorAddress("images", ...) 和 context->setOutputTensorAddress("output0", ...) 将GPU内存地址与张量名称永久绑定。
// 运行阶段: 之后每次执行推理，只需简单调用 context->enqueueV3(stream) 即可。引擎已经“记住”了数据该从哪里读、结果该往哪里写。
// 优点: 减少了每次调用的参数传递，API更清晰，也可能带来微小的性能提升。
// 对您C++代码的影响 (“静默失败”的原因):
// 您的C++代码在主循环前调用了 setInputTensorAddress 和 setOutputTensorAddress，这符合新版API的规范，这一步是正确的。
// 但是，在主循环内调用的 armor_context->enqueueV3(detect_stream) 很可能是一个被rm库封装过的、内部实现不正确的函数，或者您使用的enqueueV3版本与新API的期望不符。
// 新版enqueueV3期望地址已被预先绑定，所以它不再接受一个bindings数组。如果您的代码（或库）仍在尝试以旧方式调用它，或者在创建上下文时没有正确地完成所有设置，TensorRT引擎就不会报错，因为它收到了一个合法的“执行”指令。但由于它不知道具体要在哪个内存地址上工作（或者配置是无效的），它实际上什么也没做，最终导致您从GPU拷贝回来的输出缓冲区里全是初始的零。

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <chrono>
#include <thread>
#include <unistd.h> // 用于 access() 函数检查文件是否存在

// -------------------- 依赖库头文件 --------------------
// 自定义库，可能包含相机、时间工具等
#include <dprm/dprm.h>
#include <dprm/cudatools.h>

// OpenCV 核心及CUDA加速模块
#include <opencv2/opencv.hpp>
#include "opencv2/cudawarping.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"
#include "opencv2/dnn.hpp" // 用于 cv::dnn::blobFromImage

// TensorRT 推理引擎头文件
#include <NvInfer.h>
#include <NvOnnxParser.h>

// -------------------- CUDA 错误检查宏 --------------------
// 定义一个宏，用于检查所有CUDA API调用的返回值
// 如果调用失败，则打印详细的错误信息（文件名、行号、错误描述）并退出程序
#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                        << " - " << cudaGetErrorString(err) << std::endl;  \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)


// ====================================================================================
// 1. 参数结构体 (AppParams)
//    - 功能: 将所有可配置的应用程序参数集中到一个结构体中，方便管理和传递。
// ====================================================================================
struct AppParams {
    // --- 模型相关参数 ---
    std::string onnx_path;          // ONNX模型文件的路径 (必需)
    int infer_width       = 416;    // 模型推理所需的输入图像宽度
    int infer_height      = 416;    // 模型推理所需的输入图像高度
    int class_num         = 14;     // 模型可以识别的类别总数
    int bboxes_num        = 10647;  // YOLO模型输出的预测框总数 (例如 3x(13x13+26x26+52x52))
    double conf_thresh    = 0.4;    // 置信度阈值，低于此值的预测框将被过滤
    double nms_thresh     = 0.5;    // 非极大值抑制(NMS)的IOU阈值

    // --- 相机相关参数 ---
    double exposure       = 2500.0; // 相机曝光时间 (单位: 微秒)
    double gain           = 12.0;   // 相机增益
    double gamma          = 200.0;  // 相机Gamma值 (注意: DPRM库中gamma的范围通常是0-255)
};

// ====================================================================================
// 2. 帮助与命令行解析函数
// ====================================================================================

/**
 * @brief 打印程序的用法说明
 * @param prog_name 程序的可执行文件名 (argv[0])
 * @param defaults 默认参数实例，用于显示默认值
 */
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

/**
 * @brief 解析命令行传入的参数，并更新AppParams结构体
 * @param argc 参数数量
 * @param argv 参数值数组
 * @param params 用于存储解析结果的AppParams对象的引用
 */
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

/**
 * @brief 在图像上绘制检测到的边界框和标签
 * @param image 要绘制的图像 (cv::Mat)
 * @param bboxes 包含检测结果的 YoloRect 向量
 */
// 定义类别名称，顺序需要与模型训练时一致
const std::vector<std::string> CLASS_NAMES = {"B1","B2","B3","B4","B5","BHero","R1","R2","R3","R4","R5","RHero","RQS","BQS"};
void draw_bboxes(cv::Mat& image, const std::vector<rm::YoloRect>& bboxes) {
    for (const auto& box : bboxes) {
        // 安全检查，防止类别ID越界
        if (box.class_id >= CLASS_NAMES.size()) continue;
        // 绘制矩形框
        cv::rectangle(image, box.box, cv::Scalar(0, 255, 0), 2);
        // 准备标签文本，包含类别名和置信度
        std::string label = CLASS_NAMES[box.class_id] + ": " + cv::format("%.2f", box.confidence);
        // 计算文本尺寸以便绘制背景
        int baseline;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        // 绘制文本背景框
        cv::rectangle(image,
                      cv::Point(box.box.x, box.box.y - label_size.height - baseline),
                      cv::Point(box.box.x + label_size.width, box.box.y),
                      cv::Scalar(0, 255, 0),
                      cv::FILLED);
        // 放置文本
        cv::putText(image, label,
                    cv::Point(box.box.x, box.box.y - baseline),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}


// ====================================================================================
// 3. 主函数 (main)
//    - 功能: 整个应用程序的入口点和主流程控制器。
// ====================================================================================
int main(int argc, char* argv[]) {
    // 1. 初始化与参数解析
    AppParams params;
    parse_arguments(argc, argv, params);
    
    // 检查必需参数：模型路径
    if (params.onnx_path.empty()) {
        std::cerr << "错误: 必须通过 -m 或 --model 提供模型路径。" << std::endl;
        print_usage(argv[0], AppParams());
        return -1;
    }
    
    // 根据ONNX文件路径自动生成TensorRT引擎文件（.engine）的路径
    std::string engine_file = params.onnx_path;
    size_t dot_pos = engine_file.rfind(".onnx");
    if (dot_pos != std::string::npos) engine_file.replace(dot_pos, 5, ".engine");
    else engine_file += ".engine";
    
    std::cout << "ONNX model file: " << params.onnx_path << std::endl;
    std::cout << "TensorRT engine file: " << engine_file << std::endl;

    // 2. 加载 TensorRT 模型
    nvinfer1::IExecutionContext* armor_context = nullptr;
    rm::message("Loading YOLO model...", rm::MSG_NOTE);
    // 优先加载已序列化的.engine文件，如果不存在，则解析.onnx文件并生成.engine文件
    if (access(engine_file.c_str(), F_OK) == 0) { // 如果.engine文件已存在
        if (!rm::initTrtEngine(engine_file, &armor_context)) return -1;
    } else if (access(params.onnx_path.c_str(), F_OK) == 0) { // 如果.onnx文件存在
        if (!rm::initTrtOnnx(params.onnx_path, engine_file, &armor_context, 1U)) return -1;
    } else { // 如果两个文件都不存在
        rm::message("Model file not found at: " + params.onnx_path, rm::MSG_ERROR);
        return -1;
    }
    rm::message("YOLO model loaded successfully.", rm::MSG_OK);

    // 3. 分配CUDA内存并设置Stream
    // 定义模型输出的每个预测框的结构大小
    const int LOCATE_NUM = 4; // 边界框坐标 (x, y, w, h)
    // 您可以保留COLOR_NUM的定义，但不要在计算中使用它，或者直接删掉它
    const int COLOR_NUM = 1;
    size_t yolo_struct_size = sizeof(float) * (LOCATE_NUM + 1 + params.class_num); // 移除了 "+ COLOR_NUM"
    
    // --- 分配内存 ---
    float* armor_output_host_buffer = nullptr;   // 主机(CPU)内存，用于存放从GPU传回的推理结果
    void* armor_output_device_buffer = nullptr;  // 设备(GPU)内存，用于存放模型的原始输出
    void* armor_input_device_buffer = nullptr;   // 设备(GPU)内存，用于存放预处理后送入模型的图像数据

    // 在主机上分配用于后处理的缓冲区
    armor_output_host_buffer = new float[params.bboxes_num * (yolo_struct_size / sizeof(float))];
    
    // 在设备上分配输入和输出缓冲区
    CUDA_CHECK(cudaMalloc(&armor_input_device_buffer, 3 * params.infer_width * params.infer_height * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&armor_output_device_buffer, params.bboxes_num * yolo_struct_size));
    
    // 创建CUDA Stream，用于异步执行CUDA操作
    cv::cuda::Stream cv_stream;
    cudaStream_t detect_stream = cv::cuda::StreamAccessor::getStream(cv_stream);
    
    // 4. 绑定TensorRT输入输出缓冲区
    // 将上一步分配的设备内存地址与模型中定义的输入/输出张量名称绑定起来
    const char* input_name = "images";    // 模型输入张量的名称
    const char* output_name = "output0";  // 模型输出张量的名称

    if (!armor_context->setInputTensorAddress(input_name, armor_input_device_buffer)) {
        rm::message("Failed to bind input tensor address.", rm::MSG_ERROR);
        return -1;
    }
    if (!armor_context->setOutputTensorAddress(output_name, armor_output_device_buffer)) {
        rm::message("Failed to bind output tensor address.", rm::MSG_ERROR);
        return -1;
    }
    rm::message("TensorRT buffers bound successfully.", rm::MSG_OK);

    // 5. 初始化 HIK 相机
    rm::message("Initializing HIK camera...", rm::MSG_NOTE);
    int camera_num = -1;
    // 检查可用相机数量
    if (!rm::getHikCameraNum(camera_num) || camera_num < 1) {
        rm::message("Failed to get camera or no camera found.", rm::MSG_ERROR);
        return -1;
    }

    rm::Camera* camera = new rm::Camera();
    // 使用从命令行或默认值获取的参数打开相机
    if (!rm::openHik(camera, 1, nullptr, nullptr, nullptr, params.exposure, params.gain, params.gamma)) {
        rm::message("Failed to open camera 1.", rm::MSG_ERROR);
        delete camera;
        return -1;
    }
    rm::message("Camera opened successfully.", rm::MSG_OK);

    // 6. 主循环 - 图像采集、预处理、推理、后处理、显示
    const std::string window_name = "YOLOv5 Detection";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 960, 720);

    // [新增] 创建一个用于显示预处理后图像的调试窗口
    const std::string debug_window_name = "Preprocessed Input (416x416)";
    cv::namedWindow(debug_window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(debug_window_name, 416, 416);

    // 在循环外创建 GpuMat 对象，避免重复分配内存
    cv::cuda::GpuMat gpu_frame, resized_gpu, float_gpu;
    auto frame_wait_tp = getTime(); // 用于检测取流超时
    while (true) {
        // --- 从相机缓冲区获取一帧图像 ---
        std::shared_ptr<rm::Frame> frame = camera->buffer->pop();
        if (frame == nullptr || !frame->image || frame->image->empty()) {
            if (getDoubleOfS(frame_wait_tp, getTime()) > 2.0) { // 超过2秒未取到帧则报错
                rm::message("Capture timeout!", rm::MSG_ERROR);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1)); // 稍作等待
            continue;
        }
        frame_wait_tp = getTime(); // 重置计时器

        // --- 图像预处理 (在GPU上完成) ---
        // 1. 将CPU的cv::Mat上传到GPU的cv::cuda::GpuMat
        gpu_frame.upload(*frame->image, cv_stream);
        // 2. 缩放图像到模型所需的尺寸
        cv::cuda::resize(gpu_frame, resized_gpu, cv::Size(params.infer_width, params.infer_height), 0, 0, cv::INTER_LINEAR, cv_stream);
        // 3. BGR to RGB颜色空间转换
        cv::cuda::cvtColor(resized_gpu, resized_gpu, cv::COLOR_BGR2RGB, 0, cv_stream);
        // 4. 将图像数据类型从 CV_8U (0-255) 转换为 CV_32F (0.0-1.0)
        resized_gpu.convertTo(float_gpu, CV_32F, 1.0/255.0, cv_stream);


        // --- [调试] 显示预处理后的图像 ---
        // 这个代码块的目的是将送入神经网络前的图像可视化，以检查预处理步骤是否正确
        {
            cv::Mat preprocessed_cpu;
            cv::cuda::GpuMat temp_gpu_for_display;

            // 1. 将归一化后的浮点数据(0-1)转换回8位无符号整型(0-255)
            float_gpu.convertTo(temp_gpu_for_display, CV_8U, 255.0, cv_stream);

            // 2. 将RGB格式转换回BGR，因为OpenCV的imshow函数默认按BGR格式显示
            cv::cuda::cvtColor(temp_gpu_for_display, temp_gpu_for_display, cv::COLOR_RGB2BGR, 0, cv_stream);

            // 3. 从GPU下载图像数据到CPU
            temp_gpu_for_display.download(preprocessed_cpu, cv_stream);

            // 4. 等待CUDA流完成下载操作，确保数据完整
            cv_stream.waitForCompletion();

            // 5. 在调试窗口中显示
            cv::imshow(debug_window_name, preprocessed_cpu);
        }

        // --- 准备模型输入数据 (Planar格式) ---
        // 将 interleaved (HWC) 格式的图像数据转换为 planar (CHW) 格式
        std::vector<cv::cuda::GpuMat> channels;
        cv::cuda::split(float_gpu, channels, cv_stream);
        // 依次将R, G, B三个通道的数据拷贝到连续的设备输入缓冲区中
        CUDA_CHECK(cudaMemcpyAsync(armor_input_device_buffer, channels[0].data, params.infer_width * params.infer_height * sizeof(float), cudaMemcpyDeviceToDevice, detect_stream));
        CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(armor_input_device_buffer) + params.infer_width * params.infer_height * sizeof(float), channels[1].data, params.infer_width * params.infer_height * sizeof(float), cudaMemcpyDeviceToDevice, detect_stream));
        CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(armor_input_device_buffer) + 2 * params.infer_width * params.infer_height * sizeof(float), channels[2].data, params.infer_width * params.infer_height * sizeof(float), cudaMemcpyDeviceToDevice, detect_stream));

        // =======================> 在这里插入下面的诊断代码 <=======================

// 【诊断开始】
{
    // 1. 创建一个临时的CPU(主机)缓冲区，大小与GPU输入缓冲区完全相同
    size_t buffer_size = 3 * params.infer_width * params.infer_height;
    float* host_input_check_buffer = new float[buffer_size];

    // 2. 将GPU上的模型输入数据，拷贝回我们刚创建的CPU缓冲区
    //    这里使用同步拷贝，确保数据传输完成后再继续
    CUDA_CHECK(cudaMemcpy(host_input_check_buffer, armor_input_device_buffer, buffer_size * sizeof(float), cudaMemcpyDeviceToHost));

    // 3. 检查这个缓冲区的内容
    std::cout << "--- Checking final model input buffer ---" << std::endl;
    
    // 检查R通道的第一个像素值 (缓冲区开头)
    std::cout << "First R pixel value: " << host_input_check_buffer[0] << std::endl;

    // 检查G通道的第一个像素值 (缓冲区 1/3 处)
    size_t green_channel_start_index = params.infer_width * params.infer_height;
    std::cout << "First G pixel value: " << host_input_check_buffer[green_channel_start_index] << std::endl;

    // 检查B通道的第一个像素值 (缓冲区 2/3 处)
    size_t blue_channel_start_index = 2 * params.infer_width * params.infer_height;
    std::cout << "First B pixel value: " << host_input_check_buffer[blue_channel_start_index] << std::endl;

    // 4. 计算整个缓冲区的总和，看它是否为零
    double total_sum = 0.0;
    for (size_t i = 0; i < buffer_size; ++i) {
        total_sum += host_input_check_buffer[i];
    }
    std::cout << "Sum of all values in input buffer: " << total_sum << std::endl;
    std::cout << "------------------------------------------" << std::endl;

    // 5. 释放临时缓冲区
    delete[] host_input_check_buffer;
}
// 【诊断结束】



        // --- 执行模型推理 ---
        // 异步执行推理
        if (!rm::detectEnqueue(armor_context, detect_stream)) {
            rm::message("TensorRT inference failed!", rm::MSG_ERROR);
            break;
        }

        // --- 获取输出 ---
        // 将推理结果从设备(GPU)拷贝回主机(CPU)的缓冲区
        rm::detectOutput(
            armor_output_host_buffer,
            static_cast<const float*>(armor_output_device_buffer), // 必须传递 float* 类型
            &detect_stream,
            yolo_struct_size,
            params.bboxes_num
        );

        //测试开始
        // 【第一步：添加显式同步】确保数据已从GPU完全拷贝到CPU
cudaStreamSynchronize(detect_stream);

// 【第二步：插入诊断代码】
{
    float max_confidence = 0.0f;
    int max_conf_box_index = -1;

    // 遍历所有10647个预测框
    for (int i = 0; i < params.bboxes_num; ++i) {
        // 计算当前框数据的起始地址
        // 每个框有 19 个 float: 4(box) + 1(conf) + 14(classes)
        float* current_box_data = armor_output_host_buffer + i * (4 + 1 + params.class_num);
        
        // 第5个元素 (索引为4) 是物体置信度
        float confidence = current_box_data[4];

        if (confidence > max_confidence) {
            max_confidence = confidence;
            max_conf_box_index = i;
        }
    }

    // 打印这一帧中找到的最高置信度
    std::cout << "Max confidence in this frame: " << max_confidence << std::endl;

    // 如果最高置信度不为0，打印该框的详细信息
    if (max_conf_box_index != -1) {
        std::cout << "--- Details for box with highest confidence ---" << std::endl;
        float* best_box_data = armor_output_host_buffer + max_conf_box_index * 19;
        std::cout << "Box Index: " << max_conf_box_index << std::endl;
        std::cout << "Raw Coords (x,y,w,h): " << best_box_data[0] << ", " << best_box_data[1] << ", " << best_box_data[2] << ", " << best_box_data[3] << std::endl;
        std::cout << "Objectness Confidence: " << best_box_data[4] << std::endl;
        std::cout << "Class Scores: ";
        for (int j = 0; j < params.class_num; ++j) {
            std::cout << best_box_data[5 + j] << " ";
        }
        std::cout << "\n---------------------------------------------" << std::endl;
    }
}

        //测试结束
        
        // --- NMS 后处理 ---
        // 在CPU上对模型的原始输出进行解析和非极大值抑制，得到最终的检测框列表
        frame->yolo_list = rm::yoloArmorNMS_V5(
            armor_output_host_buffer, params.bboxes_num, params.class_num,
            params.conf_thresh, params.nms_thresh, frame->width, frame->height, // 原始图像尺寸
            params.infer_width, params.infer_height // 推理时图像尺寸
        );
        
        // --- 绘制和显示结果 ---
        // 如果有检测结果，则在原图上绘制
        if (!frame->yolo_list.empty()) {
            draw_bboxes(*frame->image, frame->yolo_list);
        }
        // 显示处理后的图像
        cv::imshow(window_name, *frame->image);
        
        // 按 ESC 键退出循环
        if (cv::waitKey(1) == 27) {
            rm::message("User requested exit.", rm::MSG_WARNING);
            break;
        }
    }

    // 7. 释放资源
    rm::message("Releasing resources...", rm::MSG_NOTE);
    cv::destroyAllWindows(); // 关闭所有OpenCV窗口
    delete camera;           // 释放相机对象
    
    // 释放CUDA设备内存
    CUDA_CHECK(cudaFree(armor_input_device_buffer));
    CUDA_CHECK(cudaFree(armor_output_device_buffer));
    // 释放主机内存
    delete[] armor_output_host_buffer;
    
    rm::message("Cleanup complete. Exiting.", rm::MSG_NOTE);
    return 0;
}
