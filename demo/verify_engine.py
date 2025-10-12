import tensorrt as trt
import numpy as np
import cv2
import pycuda.driver as cuda
import pycuda.autoinit # 非常重要，用于初始化CUDA上下文

# 1. 预处理函数 (必须与C++代码的逻辑完全一致)
def preprocess(image_path, infer_width, infer_height):
    """读取图像，并执行和C++代码一样的预处理"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    # 1. 缩放
    resized_img = cv2.resize(img, (infer_width, infer_height), interpolation=cv2.INTER_LINEAR)
    # 2. BGR -> RGB
    rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    # 3. 归一化到 0.0 - 1.0
    float_img = rgb_img.astype(np.float32) / 255.0
    # 4. HWC -> CHW (Height, Width, Channel -> Channel, Height, Width)
    chw_img = np.transpose(float_img, (2, 0, 1))
    # 5. 增加 Batch 维度 -> NCHW
    batch_img = np.expand_dims(chw_img, axis=0)
    # 6. 确保数据在内存中是连续的
    return np.ascontiguousarray(batch_img)

# 2. 主程序
if __name__ == "__main__":
    ENGINE_PATH = "v5n416best.engine" # trtexec生成的引擎文件
    IMAGE_PATH = "test_image.png"     # C++程序保存的测试图像
    INFER_WIDTH = 416
    INFER_HEIGHT = 416
    
    # --- 初始化TensorRT ---
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    
    print(f"从 {ENGINE_PATH} 加载引擎...")
    with open(ENGINE_PATH, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    if engine is None:
        print("加载引擎失败。")
        exit()
    print("引擎加载成功。")
    
    context = engine.create_execution_context()
    
    # --- 准备输入和输出缓冲区 ---
    h_input = cuda.pagelocked_empty(engine.get_binding_shape(0).volume(), dtype=np.float32)
    h_output = cuda.pagelocked_empty(engine.get_binding_shape(1).volume(), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    
    stream = cuda.Stream()

    # --- 执行推理 ---
    print(f"正在处理图像: {IMAGE_PATH}")
    input_data = preprocess(IMAGE_PATH, INFER_WIDTH, INFER_HEIGHT)
    np.copyto(h_input, input_data.ravel())

    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

    # --- 分析输出结果 ---
    print("\n--- 推理输出分析 ---")
    output_shape = engine.get_binding_shape(1) # (1, 10647, 19)
    
    # 从扁平化的输出数组中提取所有框的置信度 (索引为4)
    confidences = h_output.reshape(output_shape)[0, :, 4]
    
    max_conf = np.max(confidences)
    
    print(f"在输出中找到的最大置信度: {max_conf:.6f}")

    if max_conf > 0.1: # 使用一个较低的阈值
        print("\n结论: 成功！引擎文件和模型有效，能够产出非零结果！")
    else:
        print("\n结论: 失败。问题可能比预想的更复杂。")