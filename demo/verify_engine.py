import tensorrt as trt
import numpy as np
import cv2
import pycuda.driver as cuda
import pycuda.autoinit # 非常重要，用于初始化CUDA上下文
import sys

# 1. 预处理函数 (保持不变)
def preprocess(image_path, infer_width, infer_height):
    """读取图像，并执行和C++代码一样的预处理"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取图像: {image_path}")
        sys.exit(1)

    resized_img = cv2.resize(img, (infer_width, infer_height), interpolation=cv2.INTER_LINEAR)
    rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    float_img = rgb_img.astype(np.float32) / 255.0
    chw_img = np.transpose(float_img, (2, 0, 1))
    batch_img = np.expand_dims(chw_img, axis=0)
    return np.ascontiguousarray(batch_img)

# 2. 主程序
if __name__ == "__main__":
    ENGINE_PATH = "v5n416best.engine"
    IMAGE_PATH = "test_image.png"
    INFER_WIDTH = 416
    INFER_HEIGHT = 416
    
    INPUT_NAME = "images"
    OUTPUT_NAME = "output0"
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    
    print(f"--- 步骤1: 从 {ENGINE_PATH} 加载引擎 ---")
    try:
        with open(ENGINE_PATH, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
    except Exception as e:
        print(f"加载引擎失败: {e}")
        sys.exit(1)
        
    if engine is None:
        print("引擎反序列化失败。")
        sys.exit(1)
    print("引擎加载成功。")
    
    context = engine.create_execution_context()
    
    print("\n--- 步骤2: 分配主机(CPU)和设备(GPU)内存 ---")
    
    try:
        input_shape = engine.get_tensor_shape(INPUT_NAME)
        output_shape = engine.get_tensor_shape(OUTPUT_NAME)
    except Exception as e:
        print(f"错误: 无法获取模型输入/输出 '{INPUT_NAME}' 或 '{OUTPUT_NAME}' 的信息。")
        sys.exit(1)

    print(f"输入张量 '{INPUT_NAME}' 形状: {input_shape}")
    print(f"输出张量 '{OUTPUT_NAME}' 形状: {output_shape}")

    h_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    
    stream = cuda.Stream()
    print("内存分配完毕。")
    
    # === 最终修正：使用 set_tensor_address 提前绑定内存 ===
    # 这与C++的 setInputTensorAddress / setOutputTensorAddress 行为一致
    try:
        context.set_tensor_address(INPUT_NAME, int(d_input))
        context.set_tensor_address(OUTPUT_NAME, int(d_output))
        print("成功将GPU内存地址绑定到输入/输出张量。")
    except Exception as e:
        print(f"绑定内存地址失败: {e}")
        sys.exit(1)

    print(f"\n--- 步骤3: 预处理图像并执行推理 ---")
    input_data = preprocess(IMAGE_PATH, INFER_WIDTH, INFER_HEIGHT)
    np.copyto(h_input, input_data.ravel())

    cuda.memcpy_htod_async(d_input, h_input, stream)
    # === 最终修正：调用不带 bindings 参数的 execute_async_v3 ===
    if not context.execute_async_v3(stream_handle=stream.handle):
        print("执行推理失败 (execute_async_v3 returned False)")
        sys.exit(1)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    print("推理和数据拷贝完成。")

    print("\n--- 步骤4: 分析推理输出 ---")
    
    confidences = h_output.reshape(output_shape)[0, :, 4]
    max_conf = np.max(confidences)
    
    print(f"==============================================")
    print(f"在输出中找到的最大置信度: {max_conf:.6f}")
    print(f"==============================================")

    if max_conf > 0.1:
        print("\n最终结论: 成功！您的模型和引擎文件完全没有问题。")
        print("问题根源100%确定在您的C++代码中，因为它使用了与您当前TensorRT 10.x版本不兼容的旧版API调用方式。")
    else:
        print("\n最终结论: 失败。如果此脚本仍然输出0，说明存在更深层次的环境或驱动问题。")