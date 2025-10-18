cmake_minimum_required(VERSION 3.15)
project(DPRM-Test)

set(CMAKE_CXX_STANDARD 17)
# add_compile_options(-g -O0 -w -fno-omit-frame-pointer -Wno-notes)

add_compile_options(-O3 -w)

# Cuda
find_package(CUDA)
if(CUDA_FOUND)
    message(STATUS "CUDA Version: ${CUDA_VERSION}")
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

# OpenCV
find_package(OpenCV 4.10.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# DPRM
find_package(DPRM REQUIRED)
include_directories(${DPRM_INCLUDE_DIRS})

message(STATUS "DPRM_INCLUDE_DIRS: ${DPRM_INCLUDE_DIRS}")
message(STATUS "DPRM_LIBS: ${DPRM_LIBS}")

find_library(YOLOV5_TENSORRT_LIB NAMES yolov5-tensorrt HINTS /usr/local/lib)

# 确保找到了库
if(NOT YOLOV5_TENSORRT_LIB)
    message(FATAL_ERROR "Could not find yolov5-tensorrt library. Ensure it is built and installed.")
endif()

# 添加 YOLOv5-TensorRT 的头文件路径
# 假设头文件安装在 /usr/local/include
include_directories(/usr/local/include/yolov5-tensorrt)


# camera
add_executable(uvc_lib uvc_lib.cpp)
target_link_libraries(uvc_lib
    ${DPRM_LIBS}
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
)

add_executable(hik_lib hik_lib.cpp)
target_link_libraries(hik_lib
    ${DPRM_LIBS}
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
)



if(CUDA_FOUND)
    add_executable(tensorrt_demo tensorrt_demo.cpp)
        target_link_libraries(tensorrt_demo
            ${DPRM_LIBS}
        )

    add_executable(yolov5Tensorrt yolov5Tensorrt.cpp)
    target_link_libraries(yolov5Tensorrt
        ${DPRM_LIBS}
        ${OpenCV_LIBS}
    )
    add_executable(yoloDetectDemo yoloDetectDemo.cpp)
    target_link_libraries(yoloDetectDemo
        ${DPRM_LIBS}
        ${OpenCV_LIBS}
    )
endif()