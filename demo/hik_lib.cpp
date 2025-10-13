#include <dprm/dprm.h>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>
#include <queue>          // 用于帧队列
#include <mutex>          // 用于线程锁
#include <condition_variable> // 用于线程通信
#include <atomic>         // 用于原子操作的标志位

// 为了让代码能够独立编译，这里提供了getTime和getDoubleOfS的示例实现。
#ifndef DUMMY_TIME_FUNCTIONS
#define DUMMY_TIME_FUNCTIONS
auto getTime() { return std::chrono::steady_clock::now(); }
double getDoubleOfS(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}
#endif

// =========== 新增：线程安全的帧队列 ===========
std::queue<cv::Mat> frame_queue;
std::mutex queue_mutex;
std::condition_variable queue_cond_var;
std::atomic<bool> terminate_writer(false); // 终止写入线程的标志
// ==========================================

// =========== 新增：视频写入线程函数 ===========
void video_writer_thread_func(const std::string& filename, int fourcc, double fps, cv::Size frame_size) {
    cv::VideoWriter writer(filename, fourcc, fps, frame_size);
    if (!writer.isOpened()) {
        rm::message("Could not open the output video file for writing in worker thread.", rm::MSG_ERROR);
        return;
    }
    rm::message("Video writer thread started.", rm::MSG_NOTE);

    while (!terminate_writer || !frame_queue.empty()) {
        cv::Mat frame_to_write;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            // 当队列为空且主线程没有发出终止信号时，等待
            queue_cond_var.wait(lock, [] { return !frame_queue.empty() || terminate_writer; });

            if (!frame_queue.empty()) {
                frame_to_write = frame_queue.front();
                frame_queue.pop();
            }
        } // unique_lock 在此自动解锁

        if (!frame_to_write.empty()) {
            writer.write(frame_to_write);
        }
    }

    writer.release();
    rm::message("Video writer thread finished and file saved.", rm::MSG_NOTE);
}
// ==========================================

int main() {
    // 获取相机数量
    int camera_num = -1;
    bool flag_camera = rm::getHikCameraNum(camera_num);
    std::vector<rm::Camera*> camera;
    camera.resize(camera_num + 1, nullptr);
    if (!flag_camera) {
        rm::message("Failed to get camera number", rm::MSG_ERROR);
        return -1;
    }
    rm::message("get camera number " + std::to_string(camera_num), rm::MSG_NOTE);

    float yaw, pitch, roll;
    int camera_index = 1;

    double exp = 2500.0;
    double gain = 12.0;
    double rate = 200.0;

    if (camera_num >= 1) {
        camera[1] = new rm::Camera();
        flag_camera = rm::openHik(camera[1], 1, &yaw, &pitch, &roll, exp, gain, rate);

        if (!flag_camera) {
            rm::message("Failed to open camera", rm::MSG_ERROR);
            return -1;
        }
    } else {
        rm::message("No cameras found.", rm::MSG_ERROR);
        return -1;
    }

    // =========== 启动视频写入线程 ===========
    const std::string output_filename = "output_video.avi";
    int frame_width = camera[1]->width;
    int frame_height = camera[1]->height;
    std::thread writer_thread(video_writer_thread_func, output_filename, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), rate, cv::Size(frame_width, frame_height));
    // =======================================

    const std::string window_name = "Camera Preview";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 800, 600);

    auto frame_wait = getTime();
    auto recording_start_time = std::chrono::steady_clock::now();
    bool is_recording = true;
    const double recording_duration_seconds = 5.0;

    while (true) {
        rm::Camera* camera_t = camera[camera_index];
        std::shared_ptr<rm::Frame> frame = camera_t->buffer->pop();

        while (frame == nullptr) {
            frame = camera_t->buffer->pop();
            if (getDoubleOfS(frame_wait, getTime()) > 0.5) {
                rm::message("Capture timeout", rm::MSG_ERROR);
                exit(-1);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        frame_wait = getTime();

        if (frame && frame->image && !frame->image->empty()) {
            
            // =========== 将帧放入队列 ===========
            if (is_recording) {
                auto current_time = std::chrono::steady_clock::now();
                std::chrono::duration<double> elapsed_seconds = current_time - recording_start_time;
                if (elapsed_seconds.count() < recording_duration_seconds) {
                    {
                        std::lock_guard<std::mutex> lock(queue_mutex);
                        // 必须克隆！因为frame->image很快会被相机sdk回收复用
                        frame_queue.push(frame->image->clone()); 
                    }
                    queue_cond_var.notify_one(); // 唤醒写入线程
                } else {
                    is_recording = false;
                    rm::message("5s recording finished. Signalling writer thread to terminate...", rm::MSG_NOTE);
                    terminate_writer = true;
                    queue_cond_var.notify_one(); // 确保写入线程能被唤醒以检查终止标志
                }
            }
            // ==========================================

            cv::Mat display_image = frame->image->clone(); // 为显示创建一个副本，避免线程冲突

            std::string info = "Camera " + std::to_string(frame->camera_id) + " | " + std::to_string(frame->width) + "x" + std::to_string(frame->height);
            cv::putText(display_image, info, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
            
            std::string pose_info = "Yaw: " + std::to_string(frame->yaw) + " | Pitch: " + std::to_string(frame->pitch) + " | Roll: " + std::to_string(frame->roll);
            cv::putText(display_image, pose_info, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 1);
            
            cv::imshow(window_name, display_image);

        } else {
            rm::message("Invalid frame received", rm::MSG_WARNING);
        }

        if (cv::waitKey(1) == 27) { // ESC键
            rm::message("User requested exit", rm::MSG_WARNING);
            break;
        }
    }

    // =========== 清理工作 ===========
    rm::message("Main loop finished. Waiting for writer thread to join...", rm::MSG_NOTE);
    terminate_writer = true;
    queue_cond_var.notify_one(); // 确保写入线程退出
    writer_thread.join(); // 等待写入线程执行完毕
    
    cv::destroyWindow(window_name);
    if (camera[1] != nullptr) {
        delete camera[1];
        camera[1] = nullptr;
    }

    return 0;
}