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
#include <openrm/openrm.h>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>
using namespace std;

int main() {
    std::vector<std::string> device_list;
    // rm::listUVC(device_list, "usb_cam_");
    rm::listUVC(device_list, "video");
    std::vector<rm::Camera*> cameras;
    std::vector<TimePoint> tp;

    for (int i = 0; i < device_list.size(); i++) {
        rm::Camera* camera = new rm::Camera();
        rm::openUVC(camera, 1920, 1080, 30, 24, device_list[i]);
        rm::runUVC(camera, nullptr, 30);
        cameras.push_back(camera);
        TimePoint tp0 = getTime();
        tp.push_back(tp0);
    }

    while (1) {
        for (int i = 0; i < cameras.size(); i++) {
            std::shared_ptr<rm::Frame> frame;
            frame = nullptr;

            frame = cameras[i]->buffer->pop();
            while (frame == nullptr) {
                frame = cameras[i]->buffer->pop();
            }

            TimePoint tp1 = getTime();

            double dt = getDoubleOfS(tp[i], tp1);
            tp[i] = tp1;

            rm::print3d(i, dt, 1 / dt, "id", "dt", "fps");
        }
    }
}
