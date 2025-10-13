#include <dprm/dprm.h>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>

// 为了让代码能够独立编译，这里提供了getTime和getDoubleOfS的示例实现。
// 如果您的项目中已有这些函数，请删除或注释掉这部分。

using namespace std;

int main() {
    // 获取相机数量
    int camera_num = -1;
    bool flag_camera = rm::getHikCameraNum(camera_num);
    std::vector<rm::Camera*> camera;
    camera.clear();
    camera.resize(camera_num + 1, nullptr);
    if (!flag_camera) {
        rm::message("Failed to get camera number", rm::MSG_ERROR);
        return -1; // main函数建议返回int
    }
    rm::message("get camera number " + std::to_string(camera_num), rm::MSG_NOTE);

    float yaw;
    float pitch;
    float roll;

    int camera_index = 1;
    // int camera_base = 1; // 未使用的变量
    // int camera_far = 1;  // 未使用的变量

    // =========== 新增：视频录制相关变量 ===========
    cv::VideoWriter video_writer;
    const std::string output_filename = "output_video.avi";
    bool is_recording_setup = false;
    // ==========================================

    double exp = 2500.0;
    double gain = 12.0;
    double rate = 200.0;

    // 初始化单相机
    if (camera_num >= 1) { // 检查至少有一个相机
        camera[1] = new rm::Camera();

        flag_camera = rm::openHik(
            camera[1], 1, &yaw, &pitch, &roll,
            exp, gain, rate);

        if (!flag_camera) {
            rm::message("Failed to open camera", rm::MSG_ERROR);
            return -1;
        }

        // =========== 新增：配置并打开VideoWriter ===========
        int frame_width = camera[1]->width;
        int frame_height = camera[1]->height;
        double fps = rate; // 使用相机设置的帧率作为视频FPS

        // 定义编码器并创建VideoWriter对象, 使用MJPG编码器生成.avi文件
        video_writer.open(output_filename, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(frame_width, frame_height));

        if (!video_writer.isOpened()) {
            rm::message("Could not open the output video file for writing.", rm::MSG_ERROR);
            return -1;
        }
        is_recording_setup = true;
        rm::message("Video writer is ready. Recording will start.", rm::MSG_NOTE);
        // =================================================

        // rm::mallocYoloCameraBuffer(&camera[1]->rgb_host_buffer, &camera[1]->rgb_device_buffer, camera[1]->width, camera[1]->height);

    } else {
        rm::message("No cameras found.", rm::MSG_ERROR);
        return -1;
    }


    const std::string window_name = "Camera Preview";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 800, 600);

    auto frame_wait = getTime();

    // =========== 新增：录制计时器设置 ===========
    auto recording_start_time = std::chrono::steady_clock::now();
    bool recording_finished = false;
    const double recording_duration_seconds = 5.0;
    // =========================================

    while (true) {
        rm::Camera* camera_t = camera[camera_index];
        std::shared_ptr<rm::Frame> frame = camera_t->buffer->pop();

        while (frame == nullptr) {
            frame = camera_t->buffer->pop();
            double delay = getDoubleOfS(frame_wait, getTime());
            if (delay > 0.5) {
                rm::message("Capture timeout", rm::MSG_ERROR);
                exit(-1);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1)); // 避免CPU空转
        }

        // 重置等待时间
        frame_wait = getTime();

        // 显示图像
        if (frame && frame->image && !frame->image->empty()) {

            // =========== 新增：视频录制逻辑 ===========
            if (is_recording_setup && !recording_finished) {
                auto current_time = std::chrono::steady_clock::now();
                std::chrono::duration<double> elapsed_seconds = current_time - recording_start_time;

                if (elapsed_seconds.count() < recording_duration_seconds) {
                    // 如果时间未到5秒，则写入帧
                    video_writer.write(*frame->image);
                } else {
                    // 时间到达5秒，释放写入器并设置标志位
                    video_writer.release();
                    recording_finished = true;
                    rm::message("Finished recording " + std::to_string(recording_duration_seconds) + "s of video to " + output_filename, rm::MSG_NOTE);
                }
            }
            // ==========================================

            // 添加帧信息 (这部分信息会一并录制到视频中)
            std::string info = "Camera " + std::to_string(frame->camera_id) +
                               " | " + std::to_string(frame->width) + "x" + std::to_string(frame->height);
            cv::putText(*frame->image, info, cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

            // 显示姿态信息
            std::string pose_info = "Yaw: " + std::to_string(frame->yaw) +
                                    " | Pitch: " + std::to_string(frame->pitch) +
                                    " | Roll: " + std::to_string(frame->roll);
            cv::putText(*frame->image, pose_info, cv::Point(10, 60),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 1);
            
            // 显示原始图像
            cv::imshow(window_name, *frame->image);

        } else {
            rm::message("Invalid frame received", rm::MSG_WARNING);
        }

        // 检查退出按键
        if (cv::waitKey(1) == 27) { // ESC键
            rm::message("User requested exit", rm::MSG_WARNING);
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // =========== 新增：程序退出前最终清理 ===========
    // 确保在程序退出前，如果仍在录制，则释放VideoWriter
    if (video_writer.isOpened()) {
        video_writer.release();
        rm::message("Video writer released on exit.", rm::MSG_NOTE);
    }
    // ============================================

    // 关闭窗口
    cv::destroyWindow(window_name);

    // 释放相机资源
    if (camera[1] != nullptr) {
        delete camera[1];
        camera[1] = nullptr;
    }

    return 0;
}