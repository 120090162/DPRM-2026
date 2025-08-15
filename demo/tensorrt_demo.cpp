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
#include <dprm.h>
#include <unistd.h>

using namespace rm;

int main() {
    nvinfer1::IExecutionContext* armor_context_;

    std::string onnx_file = "./best_cv.onnx";
    std::string engine_file = "model.engine";

    if (access(engine_file.c_str(), F_OK) == 0) {
        if (!rm::initTrtEngine(engine_file, &armor_context_))
            exit(-1);
    } else if (access(onnx_file.c_str(), F_OK) == 0) {
        if (!rm::initTrtOnnx(onnx_file, engine_file, &armor_context_, 1U))
            exit(-1);
    } else {
        rm::message("No model file found!", rm::MSG_ERROR);
        exit(-1);
    }

    return 0;
}
