#include <dprm/dprm.h>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>
using namespace std;

int main() {
    // 获取相机数量
    int camera_num = -1;
    bool flag_camera = rm::getHikCameraNum(camera_num);
    std::vector<rm::Camera*> camera;
    camera.clear();
    camera.resize(camera_num + 1, nullptr);
    if(!flag_camera) {
        rm::message("Failed to get camera number", rm::MSG_ERROR);
        return false;
    }
    rm::message("get camera number "+ std::to_string(camera_num), rm::MSG_NOTE);

    float yaw;
    float pitch;
    float roll;

    int camera_index = 1;
    int camera_base = 1;
    int camera_far = 1;

    // 初始化单相机
    if(camera_num == 1) {
        double exp = 2500.0;
        double gain = 12.0;
        double rate = 200.0;

        camera[1] = new rm::Camera();

        flag_camera = rm::openHik(
            camera[1], 1, &yaw, &pitch, &roll,
            exp, gain, rate);

        if(!flag_camera) {
            rm::message("Failed to open camera", rm::MSG_ERROR);
            return false;
        }

        // rm::mallocYoloCameraBuffer(&camera[1]->rgb_host_buffer, &camera[1]->rgb_device_buffer, camera[1]->width, camera[1]->height);

    }

    const std::string window_name = "Camera Preview";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 800, 600);

    auto frame_wait = getTime();
    while(true){
        rm::Camera* camera_t = camera[camera_index];
        std::shared_ptr<rm::Frame> frame = camera_t->buffer->pop();
        
        while(frame == nullptr) {
            frame = camera_t->buffer->pop();
            // 展示frame
            // TODO:
            double delay = getDoubleOfS(frame_wait, getTime());
            if (delay > 0.5) {
                rm::message("Capture timeout", rm::MSG_ERROR);
                exit(-1);
            }
        }

        // 重置等待时间
        frame_wait = getTime();
        
        // 显示图像
        if (frame && frame->image && !frame->image->empty()) {
            // 显示原始图像
            cv::imshow(window_name, *frame->image);
            
            // 添加帧信息
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

    // 关闭窗口
    cv::destroyWindow(window_name);
}