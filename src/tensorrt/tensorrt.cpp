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
#include "tensorrt/tensorrt.h"
#include <unistd.h>
#include <fstream>
#include <iostream>
#include "uniterm/uniterm.h"

using namespace nvinfer1;
using namespace nvonnxparser;

// These are used to define input/output tensor names,
// you can set them to whatever you want.
const static char* kInputTensorName = "data";
const static char* kOutputTensorName = "prob";

bool rm::initTrtOnnx(
    const std::string& onnx_file,
    const std::string& engine_file,
    IExecutionContext** context,
    unsigned int batch_size
) {
    try {
        // 检查ONNX文件是否存在
        if (access(onnx_file.c_str(), F_OK) != 0) {
            throw std::runtime_error("ONNX file not found.");
        }

        // 创建 TensorRT 构建器
        Logger logger;
        auto infer_builder = createInferBuilder(logger); //创建builder
        if (!infer_builder) {
            throw std::runtime_error("Failed to create TensorRT builder.");
        }

        // 使用显式批处理
        const auto explicit_batch = 1U
            << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

        // 创建网络对象，设置批处理大小
        auto network = infer_builder->createNetworkV2(explicit_batch | batch_size);
        if (!network) {
            throw std::runtime_error("Failed to create TensorRT network.");
        }

        // 创建ONNX解析器, 解析ONNX模型
        auto parser = nvonnxparser::createParser(*network, logger);
        if (!parser) {
            throw std::runtime_error("Failed to create ONNX parser.");
        }

        if (!parser->parseFromFile(onnx_file.c_str(), static_cast<int>(ILogger::Severity::kINFO))) {
            throw std::runtime_error("Failed to parse ONNX model.");
        }

        // 创建推理引擎
        // 创建config，用于定义一系列生成engine相关的设置，例如最大显存占用，浮点精度等
        auto config = infer_builder->createBuilderConfig();
        if (!config) {
            throw std::runtime_error("Failed to create TensorRT builder config.");
        }
        if (infer_builder->platformHasFastFp16()) {
            config->setFlag(BuilderFlag::kFP16);
        }

        // 创建推理引擎
        // auto engine = infer_builder->buildEngineWithConfig(*network, *config);
        // if (!engine) {
        //     throw std::runtime_error("Failed to build TensorRT engine.");
        // }

        // 创建推理上下文
        // *context = engine->createExecutionContext();
        // if (!(*context)) {
        //     throw std::runtime_error("Failed to create TensorRT execution context.");
        // }

        // 保存推理引擎到文件
        // std::ofstream file(engine_file, std::ios::binary);
        // if (!file) {
        //     throw std::runtime_error("Failed to open engine file for writing.");
        // }
        // file.write(
        //     reinterpret_cast<const char*>(engine->serialize()->data()),
        //     engine->serialize()->size()
        // );
        // file.close();

        // 释放资源
        // parser->destroy();
        // network->destroy();
        // infer_builder->destroy();
        // config->destroy();
        // engine->destroy();

        // 使用 buildSerializedNetwork 替代 buildEngineWithConfig
        IHostMemory* serialized_engine = infer_builder->buildSerializedNetwork(*network, *config);
        if (!serialized_engine)
            throw std::runtime_error("Failed to build TensorRT engine.");

        // 保存引擎文件
        std::ofstream file(engine_file, std::ios::binary);
        if (!file)
            throw std::runtime_error("Failed to open engine file for writing.");
        file.write(
            reinterpret_cast<const char*>(serialized_engine->data()),
            serialized_engine->size()
        );
        file.close();

        // 创建运行时并反序列化引擎
        auto runtime = createInferRuntime(logger);
        if (!runtime)
            throw std::runtime_error("Failed to create runtime.");

        auto engine =
            runtime->deserializeCudaEngine(serialized_engine->data(), serialized_engine->size());
        if (!engine)
            throw std::runtime_error("Failed to deserialize engine.");

        // 创建执行上下文
        *context = engine->createExecutionContext();
        if (!(*context))
            throw std::runtime_error("Failed to create execution context.");

        // 释放资源（使用 delete 替代 destroy）
        delete parser; // 替代 parser->destroy()
        delete network; // 替代 network->destroy()
        delete config; // 替代 config->destroy()
        delete serialized_engine; // IHostMemory 仍需 destroy
        delete infer_builder; // 替代 infer_builder->destroy()

        // 注意：engine 和 runtime 由执行上下文管理，不能在此销毁
        rm::message("TensorRT ONNX model parsed and engine built", rm::MSG_OK);
        return true;

    } catch (const std::exception& e) {
        std::string error_message = e.what();
        rm::message("TensoRT ONNX : " + error_message, rm::MSG_ERROR);

        // if (*context) {
        //     (*context)->destroy();
        // }

        if (*context) {
            delete *context; // 替代 destroy()
            *context = nullptr;
        }

        return false;
    }
}

bool rm::initTrtEngine(const std::string& engine_file, nvinfer1::IExecutionContext** context) {
    try {
        // 检查引擎文件是否存在
        if (access(engine_file.c_str(), F_OK) != 0) {
            throw std::runtime_error("Engine file not found.");
        }

        // 读取引擎文件
        std::ifstream file(engine_file, std::ios::binary);
        if (!file.good()) {
            throw std::runtime_error("Failed to open engine file for reading.");
        }

        size_t size = 0; // 存储二进制文件字符的数量
        file.seekg(0, std::ios::end); //将文件指针移动到文件末尾
        size = file.tellg(); //获取当前文件指针的位置，即文件的大小
        file.seekg(0, std::ios::beg); //文件指针移回文件开始处
        char* serialized_engine = new char[size]; //分配足够的内存储存文件内容
        assert(serialized_engine); //检查内存是否分配成功
        file.read(serialized_engine, size); //读取文件信息，并存储在trtModelStream
        file.close(); //关闭文件

        // 创建 TensorRT 运行时
        Logger logger;
        auto runtime = createInferRuntime(logger);
        if (!runtime) {
            throw std::runtime_error("Failed to create TensorRT runtime.");
        }

        // 反序列化引擎
        auto engine = runtime->deserializeCudaEngine(serialized_engine, size);
        if (!engine) {
            throw std::runtime_error("Failed to deserialize TensorRT engine.");
        }

        // 创建推理上下文
        *context = engine->createExecutionContext();
        if (!(*context)) {
            throw std::runtime_error("Failed to create TensorRT execution context.");
        }

        // 释放资源
        delete[] serialized_engine;

        rm::message("TensorRT Engine OK", rm::MSG_OK);
        return true;

    } catch (const std::exception& e) {
        std::string error_message = e.what();
        rm::message("TensoRT Engine : " + error_message, rm::MSG_ERROR);
        // if (*context) {
        //     (*context)->destroy();
        // }
        if (*context) {
            delete *context; // 替代 destroy()
            *context = nullptr;
        }
        return false;
    }
}

bool rm::initCudaStream(cudaStream_t* stream) {
    try {
        // 创建CUDA流
        cudaSetDevice(0);
        cudaStreamCreate(stream);
        rm::message("CUDA Stream created", rm::MSG_OK);
        return true;

    } catch (const std::exception& e) {
        std::string error_message = e.what();
        rm::message("CUDA Stream : " + error_message, rm::MSG_ERROR);
        return false;
    }
}

// void rm::detectEnqueue(
//     float* input_device_buffer,
//     float* output_device_buffer,
//     nvinfer1::IExecutionContext** context,
//     cudaStream_t* stream
// ) {
//     float* device_buffer[2];
//     device_buffer[0] = input_device_buffer;
//     device_buffer[1] = output_device_buffer;
//     (*context)->enqueueV2((void**)device_buffer, *stream, nullptr);
// }

void rm::detectEnqueue(
    float* input_device_buffer,
    float* output_device_buffer,
    nvinfer1::IExecutionContext** context,
    cudaStream_t* stream
) {
    // input, output
    (*context)->setInputTensorAddress(kInputTensorName, input_device_buffer);
    (*context)->setOutputTensorAddress(kOutputTensorName, output_device_buffer);
    (*context)->enqueueV3(*stream);
    // cudaStreamSynchronize(*stream);
}

void rm::detectOutput(
    float* output_host_buffer,
    const float* output_device_buffer,
    cudaStream_t* stream,
    size_t output_struct_size,
    int bboxes_num,
    int batch_size
) {
    size_t output_size = (output_struct_size * bboxes_num + 1) * batch_size;
    cudaMemcpyAsync(
        output_host_buffer,
        output_device_buffer,
        output_size,
        cudaMemcpyDeviceToHost,
        *stream
    );
    cudaStreamSynchronize(*stream);
}

void rm::detectOutputClassify(
    float* output_host_buffer,
    const float* output_device_buffer,
    cudaStream_t* stream,
    int class_num
) {
    cudaMemcpyAsync(
        output_host_buffer,
        output_device_buffer,
        class_num * sizeof(float),
        cudaMemcpyDeviceToHost,
        *stream
    );
    cudaStreamSynchronize(*stream);
}

void rm::mallocYoloCameraBuffer(
    uint8_t** rgb_host_buffer,
    uint8_t** rgb_device_buffer,
    int rgb_width,
    int rgb_height,
    int batch_size,
    int channels
) {
    cudaMallocHost(
        reinterpret_cast<void**>(rgb_host_buffer),
        rgb_width * rgb_height * channels * batch_size * sizeof(uint8_t)
    );
    cudaMalloc(
        reinterpret_cast<void**>(rgb_device_buffer),
        rgb_width * rgb_height * channels * batch_size * sizeof(uint8_t)
    );
    rm::message("Yolo Camera Buffer allocated", rm::MSG_OK);
}

void rm::mallocYoloDetectBuffer(
    float** input_device_buffer,
    float** output_device_buffer,
    float** output_host_buffer,
    int input_width,
    int input_height,
    size_t output_struct_size,
    int bboxes_num,
    int batch_size,
    int channels
) {
    cudaMalloc(
        reinterpret_cast<void**>(input_device_buffer),
        input_width * input_height * channels * batch_size * sizeof(float)
    );
    cudaMalloc(
        reinterpret_cast<void**>(output_device_buffer),
        (output_struct_size * bboxes_num + 1) * batch_size
    );
    cudaMallocHost(
        reinterpret_cast<void**>(output_host_buffer),
        (output_struct_size * bboxes_num + 1) * batch_size
    );
    rm::message("Yolo Detect Buffer allocated", rm::MSG_OK);
}

void rm::mallocClassifyBuffer(
    float** input_host_buffer,
    float** input_device_buffer,
    float** output_device_buffer,
    float** output_host_buffer,
    int input_width,
    int input_height,
    int class_num,
    int channels
) {
    size_t input_num = input_width * input_height * channels;
    size_t input_size = input_num * sizeof(float);
    size_t output_size = class_num * sizeof(float);

    cudaMallocHost(reinterpret_cast<void**>(input_host_buffer), input_size);
    cudaMalloc(reinterpret_cast<void**>(input_device_buffer), input_size);
    cudaMalloc(reinterpret_cast<void**>(output_device_buffer), output_size);
    cudaMallocHost(reinterpret_cast<void**>(output_host_buffer), output_size);
    rm::message("Classify Buffer allocated", rm::MSG_OK);
}

void rm::freeYoloCameraBuffer(uint8_t* rgb_host_buffer, uint8_t* rgb_device_buffer) {
    cudaFreeHost(rgb_host_buffer);
    cudaFree(rgb_device_buffer);
    rm::message("Yolo Camera Buffer free", rm::MSG_WARNING);
}

void rm::freeYoloDetectBuffer(
    float* input_device_buffer,
    float* output_device_buffer,
    float* output_host_buffer
) {
    cudaFree(input_device_buffer);
    cudaFree(output_device_buffer);
    cudaFreeHost(output_host_buffer);
    rm::message("Yolo Detect Buffer freed", rm::MSG_WARNING);
}

void rm::freeClassifyBuffer(
    float* input_host_buffer,
    float* input_device_buffer,
    float* output_device_buffer,
    float* output_host_buffer
) {
    cudaFreeHost(input_host_buffer);
    cudaFree(input_device_buffer);
    cudaFree(output_device_buffer);
    cudaFreeHost(output_host_buffer);
    rm::message("Classify Buffer freed", rm::MSG_WARNING);
}

void rm::memcpyYoloCameraBuffer(
    uint8_t* rgb_mat_data,
    uint8_t* rgb_host_buffer,
    uint8_t* rgb_device_buffer,
    int rgb_width,
    int rgb_height,
    int channels
) {
    size_t rgb_size = rgb_width * rgb_height * channels * sizeof(uint8_t);
    memcpy(rgb_host_buffer, rgb_mat_data, rgb_size);
    cudaMemcpy(rgb_device_buffer, rgb_host_buffer, rgb_size, cudaMemcpyHostToDevice);
}

void rm::memcpyClassifyBuffer(
    uint8_t* mat_data,
    float* input_host_buffer,
    float* input_device_buffer,
    int input_width,
    int input_height,
    int channels
) {
    size_t input_num = input_width * input_height * channels;
    size_t input_size = input_num * sizeof(float);

    if (channels == 3) {
        int pixel_num = input_width * input_height;
        for (size_t i = 0; i < pixel_num; i++) {
            input_host_buffer[i] = static_cast<float>(mat_data[i * channels + 2]);
            input_host_buffer[i + pixel_num] = static_cast<float>(mat_data[i * channels + 1]);
            input_host_buffer[i + pixel_num * 2] = static_cast<float>(mat_data[i * channels]);
        }
    } else {
        for (size_t i = 0; i < input_num; i++) {
            input_host_buffer[i] = static_cast<float>(mat_data[i]);
        }
    }
    cudaMemcpy(input_device_buffer, input_host_buffer, input_size, cudaMemcpyHostToDevice);
}
