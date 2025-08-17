/*
 * Copyright (c) 2026, Cuhksz DragonPass. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <thread>
#include "structure/camera.hpp"
#include "uniterm/uniterm.h"
#include "video/video.h"
#include "video/hik/MvCameraControl.h"
#include <algorithm>

using namespace rm;
using namespace std;

static MV_CC_DEVICE_INFO_LIST device_list;
static std::map<int, void *> camap;
static int n_ret_;
struct HikParam {
    std::vector<uint8_t> data;     // 图像数据
    float* yaw;
    float* pitch;
    float* roll;
    HikParam() = default;
};

static void capture(rm::Camera* camera, unique_ptr<HikParam> callback_param, unique_ptr<MV_CC_PIXEL_CONVERT_PARAM> convert_param, double fps) {
    int delay = 1000.0 / fps;
    int fail_count_ = 0;
    int max_tries = 100;
    float yaw = 0, pitch = 0, roll = 0;
    void *device_handle_ = camap[camera->camera_id];
    int status;
    MV_FRAME_OUT out_frame;
    rm::message("Publishing image of ", camera->camera_id);
    
    TimePoint last_time = getTime();
    while (true) {
        status = MV_CC_GetImageBuffer(device_handle_, &out_frame, 1000);

        if (MV_OK == status) {
            // 像素格式转换
            // 包含：图像宽，图像高，输入数据缓存，输入数据大小，源像素格式，
            // 目标像素格式，输出数据缓存，提供的输出缓冲区大小
            convert_param->pDstBuffer = callback_param->data.data();
            convert_param->nDstBufferSize = callback_param->data.size();
            convert_param->pSrcData = out_frame.pBufAddr;
            convert_param->nSrcDataLen = out_frame.stFrameInfo.nFrameLen;
            convert_param->enSrcPixelType = out_frame.stFrameInfo.enPixelType;

            MV_CC_ConvertPixelType(device_handle_, convert_param.get());

            callback_param->data.resize(out_frame.stFrameInfo.nWidth * out_frame.stFrameInfo.nHeight * 3);

            MV_CC_FreeImageBuffer(device_handle_, &out_frame);
        } else {
            rm::message("Get buffer failed! nRet: " + to_string(n_ret_), rm::MSG_WARNING);
            MV_CC_StopGrabbing(device_handle_);
            MV_CC_StartGrabbing(device_handle_);
            fail_count_++;
            if (fail_count_ > max_tries) {
                rm::message("HikCamera failed!", rm::MSG_ERROR);
                break;
            }
            continue;
        }

        if (callback_param->yaw != nullptr && callback_param->pitch != nullptr
            && callback_param->roll != nullptr)
        {
            yaw = *(callback_param->yaw);
            pitch = *(callback_param->pitch);
            roll = *(callback_param->roll);
        }

        std::shared_ptr<rm::Frame> frame = std::make_shared<rm::Frame>();
        cv::Mat image_rgb =
            cv::Mat(camera->height, camera->width, CV_8UC3, const_cast<uint8_t*>(callback_param->data.data()));
        cv::Mat image_bgr;
        try {
            cv::cvtColor(image_rgb, image_bgr, cv::COLOR_RGB2BGR);
        } catch (const cv::Exception& e) {
            std::string error_msg = e.what();
            rm::message("Video HikCamera: cvt error at" + error_msg, rm::MSG_ERROR);
            continue;
        } catch (...) {
            rm::message("Video HikCamera: cvt error", rm::MSG_ERROR);
            continue;
        }

        TimePoint time_stamp = getTime();
        frame->time_point = time_stamp;
        frame->camera_id = camera->camera_id;
        frame->width = camera->width;
        frame->height = camera->height;
        frame->yaw = yaw;
        frame->pitch = pitch;
        frame->roll = roll;
        frame->image = std::make_shared<cv::Mat>(image_bgr);

        camera->buffer->push(frame);

        int sleep_time = delay - static_cast<int>(getNumOfMs(last_time, getTime()));
        if (sleep_time > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
        last_time = getTime();
    }
}

bool rm::getHikCameraNum(int& num, int timeout_ms, int max_tries) {
    int try_count = 0;
    // enum device
    while (try_count++ < max_tries) {
      n_ret_ = MV_CC_EnumDevices(MV_USB_DEVICE, &device_list);
      if (n_ret_ != MV_OK) {
        rm::message("Failed to enumerate devices, retrying...", rm::MSG_ERROR);
        return false;
      } else if (device_list.nDeviceNum == 0) {
        rm::message("No camera found, retrying...", rm::MSG_WARNING);
      } else {
        num = static_cast<int>(device_list.nDeviceNum);
        rm::message("Found camera count = ", num);
        return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(timeout_ms));
    }
    return false;
}

bool rm::setHikArgs(Camera* camera, 
                    double gain,
                    double acquisition_frame_rate,
                    double exposure_time,
                    const std::string& adc_bit_depth,
                    const std::string& pixel_format) 
{
    void *device_handle_ = camap[camera->camera_id];
    MVCC_FLOATVALUE f_value;

    // 设置帧率
    rm::message("Acquisition frame rate in Hz", rm::MSG_NOTE);
    MV_CC_GetFloatValue(device_handle_, "AcquisitionFrameRate", &f_value);
    double clamped_frame_rate = clamp(
        acquisition_frame_rate, 
        static_cast<double>(f_value.fMin), 
        static_cast<double>(f_value.fMax)
    );
    MV_CC_SetBoolValue(device_handle_, "AcquisitionFrameRateEnable", true);
    MV_CC_SetFloatValue(device_handle_, "AcquisitionFrameRate", clamped_frame_rate);
    rm::message("Acquisition frame rate: " + to_string(clamped_frame_rate), rm::MSG_NOTE);
    
    // 设置曝光时间
    rm::message("Exposure time in microseconds", rm::MSG_OK);
    MV_CC_GetFloatValue(device_handle_, "ExposureTime", &f_value);    
    double clamped_exposure = clamp(
        exposure_time, 
        static_cast<double>(f_value.fMin), 
        static_cast<double>(f_value.fMax)
    );
    MV_CC_SetFloatValue(device_handle_, "ExposureTime", clamped_exposure);
    rm::message("Exposure time: " + to_string(clamped_exposure), rm::MSG_NOTE);
    
    // 设置增益
    MV_CC_GetFloatValue(device_handle_, "Gain", &f_value);    
    double clamped_gain = clamp(
        gain, 
        static_cast<double>(f_value.fMin), 
        static_cast<double>(f_value.fMax)
    );
    MV_CC_SetFloatValue(device_handle_, "Gain", clamped_gain);
    rm::message("Gain: " + to_string(clamped_gain), rm::MSG_NOTE);
    
    // 设置ADC位深度
    rm::message("ADC Bit Depth, Supported values: Bits_8, Bits_12", rm::MSG_OK);
    n_ret_ = MV_CC_SetEnumValueByString(device_handle_, "ADCBitDepth", adc_bit_depth.c_str());
    if (n_ret_ == MV_OK){
        rm::message("ADC Bit Depth set to " + adc_bit_depth, rm::MSG_OK);
    }else{
        rm::message("Failed to set ADC Bit Depth", rm::MSG_ERROR);
        return false;
    }
    
    // 设置像素格式
    rm::message("Pixel Format", rm::MSG_OK);
    n_ret_ = MV_CC_SetEnumValueByString(device_handle_, "PixelFormat", pixel_format.c_str());
    if (n_ret_ == MV_OK){
        rm::message("Pixel Format set to " + pixel_format, rm::MSG_OK);
    }else{
        rm::message("Failed to set Pixel Format", rm::MSG_ERROR);
        return false;
    }
    
    return true;
}

bool rm::openHik(
    Camera* camera,
    int device_num,
    float* yaw_ptr,
    float* pitch_ptr,
    float* roll_ptr,
    double exposure,
    double gain,
    double fps
) {
    MV_IMAGE_BASIC_INFO img_info_;

    // 初始化camera对象
    if (camera == nullptr) {
        rm::message("Video Hik error at nullptr camera", rm::MSG_ERROR);
        return false;
    }
    if (camera->buffer != nullptr) {
        delete camera->buffer;
    }
    camera->buffer = new SwapBuffer<Frame>();
    camera->camera_id = device_num;

    // 打开设备
    void *camera_handle_;
    n_ret_ = MV_CC_CreateHandle(&camera_handle_, device_list.pDeviceInfo[device_num - 1]);
    if (n_ret_ != MV_OK) {
        rm::message("Failed to create camera handle!", rm::MSG_ERROR);
        return false;
    }

    n_ret_ = MV_CC_OpenDevice(camera_handle_);
    if (n_ret_ != MV_OK) {
        rm::message("Failed to open camera device!", rm::MSG_ERROR);
        return false;
    }

    // Get camera information
    n_ret_ = MV_CC_GetImageInfo(camera_handle_, &img_info_);
    if (n_ret_ != MV_OK) {
        rm::message("Failed to get camera image info!", rm::MSG_ERROR);
        return false;
    }

    // Init convert param
    unique_ptr<MV_CC_PIXEL_CONVERT_PARAM> convert_param_ = make_unique<MV_CC_PIXEL_CONVERT_PARAM>();
    convert_param_->nWidth = img_info_.nWidthValue;
    convert_param_->nHeight = img_info_.nHeightValue;
    // 设置包大小
    convert_param_->enDstPixelType = PixelType_Gvsp_RGB8_Packed;

    // 将设备句柄存入map
    camap[device_num] = camera_handle_;

    // 获取图像尺寸
    camera->width = static_cast<int>(convert_param_->nWidth);
    camera->height = static_cast<int>(convert_param_->nHeight);

    // 设置相机参数
    rm::setHikArgs(camera, gain, fps, exposure);

    // 设置图像采集参数
    unique_ptr<HikParam> callback_param_ = make_unique<HikParam>();
    callback_param_->yaw = yaw_ptr;
    callback_param_->pitch = pitch_ptr;
    callback_param_->roll = roll_ptr;
    callback_param_->data.reserve(img_info_.nHeightMax * img_info_.nWidthMax * 3);

    // 开始图像采集
    n_ret_ = MV_CC_StartGrabbing(camera_handle_);
    if (n_ret_ != MV_OK) {
        rm::message("Video Hik start failed", rm::MSG_ERROR);
        closeHik();
        return false;
    }
    rm::message("Video Hik opened", rm::MSG_OK);

    // 开始图像线程
    MV_FRAME_OUT out_frame;
    std::thread capture_thread_(&capture, camera, std::move(callback_param_), std::move(convert_param_), fps);
    capture_thread_.detach();
    return true;
}

bool rm::closeHik() {
    for (auto it = camap.begin(); it != camap.end(); it++) {
        void *device = it->second;
        if (device) {
            MV_CC_StopGrabbing(device);
            MV_CC_CloseDevice(device);
            MV_CC_DestroyHandle(&device);
        }else{
            rm::message("Video Hik close device failed", rm::MSG_ERROR);
            return false;
        }
    }
    rm::message("Video Hik closed", rm::MSG_WARNING);
    return true;
}