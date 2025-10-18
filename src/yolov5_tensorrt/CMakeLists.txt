add_library(dprm_yolov5_tensorrt
    yolov5_builder.cpp
    yolov5_common.cpp
    yolov5_detection.cpp
    yolov5_detector.cpp
    yolov5_detector_internal.cpp
    yolov5_logging.cpp
)

target_link_libraries(dprm_yolov5_tensorrt
    PRIVATE
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        nvinfer
)

target_include_directories(dprm_yolov5_tensorrt
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
        $<INSTALL_INTERFACE:include/dprm>
)

# set_target_properties(dprm_yolov5_tensorrt PROPERTIES
#     CXX_STANDARD 20
#     POSITION_INDEPENDENT_CODE ON
# )

# if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
#     target_compile_options(dprm_yolov5_tensorrt PRIVATE -W -Wall -Wno-unused-parameter)
# endif()
