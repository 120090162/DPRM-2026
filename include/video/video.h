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
#ifndef __DPRM_VIDEO_VIDEO_H__
#define __DPRM_VIDEO_VIDEO_H__
#include <cstdint>
#include <string>
#include <structure/camera.hpp>
#include <structure/stamp.hpp>
#include <vector>

namespace rm {

bool getDaHengCameraNum(int& device_num);

bool setDaHengArgs(Camera* camera, double exposure, double gain, double fps);

bool openDaHeng(
    Camera* camera,
    int device_num = 1,
    float* yaw = nullptr,
    float* pitch = nullptr,
    float* roll = nullptr,
    bool flip = false,
    double exposure = 2000.0,
    double gain = 15.0,
    double fps = 200.0
);

bool closeDaHeng();

bool testUVC(std::string& device_name);

bool listUVC(std::vector<std::string>& device_list, std::string base_name = "video");

bool openUVC(
    Camera* camera,
    unsigned int width = 1920,
    unsigned int height = 1080,
    unsigned int fps = 60,
    unsigned int buffer_num = 8,
    std::string device_name = "/dev/video0"
);

bool setUVC(
    Camera* camera,
    int exposure,
    int brightness,
    int contrast,
    int gamma,
    int gain,
    int sharpness,
    int backlight
);

bool runUVC(Camera* camera, Locate* locate_ptr, int fps);
bool closeUVC(Camera* camera);

bool getHikCameraNum(int& num, int timeout_ms = 500, int max_tries = 5);
bool setHikArgs(Camera* camera,
                    double gain,
                    double acquisition_frame_rate = 165.0,
                    double exposure_time = 5000,
                    const std::string& adc_bit_depth = "Bits_8",
                    const std::string& pixel_format = "BayerRG8");

bool openHik(
    Camera* camera,
    int device_num = 1,
    float* yaw = nullptr,
    float* pitch = nullptr,
    float* roll = nullptr,
    double exposure = 2000.0,
    double gain = 15.0,
    double fps = 200.0
);
bool closeHik();

} // namespace rm

#endif
