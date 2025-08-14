cmake_minimum_required(VERSION 3.15)
project(DPRM)

string(TIMESTAMP BUILD_TIMESTAMP "%Y-%m-%d %H:%M:%S")
message("Build timestamp is ${BUILD_TIMESTAMP}")

set(CMAKE_CXX_STANDARD 20)
# Export compile commands for clang-tidy
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set(CMAKE_BUILD_TYPE DEBUG)
# add_compile_options(-g -O0 -w -fno-omit-frame-pointer -Wno-notes)

set(CMAKE_BUILD_TYPE RELEASE)
add_compile_options(-O3 -w) # 禁用警告

# ref to https://github.com/xizhibei/blog/issues/145
# 使用ccache提高编译速度
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    message(STATUS "Set up ccache ...")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif()

# 开启 Interprocedural optimization
include(CheckIPOSupported)
check_ipo_supported(RESULT _IsIPOSupported)
    if(_IsIPOSupported)
    message(STATUS "Turn on INTERPROCEDURAL_OPTIMIZATION")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
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

# CUDA
# CUDA的库寻找需要比TensorRT早，避免精度损失，参考: https://www.cnblogs.com/yanghailin/p/14651931.html
find_package(CUDA)
if(CUDA_FOUND)
    message(STATUS "CUDA Version: ${CUDA_VERSION}")
    include_directories(/usr/local/cuda/include)
endif()

# 设置主版本号、次版本号和补丁版本号
set(DPRM_VERSION_MAJOR 0)
set(DPRM_VERSION_MINOR 0)
set(DPRM_VERSION_PATCH 1)
message("Version is ${DPRM_VERSION_MAJOR}.${DPRM_VERSION_MINOR}.${DPRM_VERSION_PATCH}")

# 生成完整的版本号
set(DPRM_VERSION ${DPRM_VERSION_MAJOR}.${DPRM_VERSION_MINOR}.${DPRM_VERSION_PATCH})

# 将宏定义添加到项目
add_definitions(-DDPRM_VERSION="${DPRM_VERSION}")

# 将版本号定义为预定义宏，使其在程序内部可访问
# 复制文件
configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/dp_version.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/dp_version.h")


# 添加子目录
add_subdirectory(src/attack)
add_subdirectory(src/kalman)
add_subdirectory(src/pointer)
add_subdirectory(src/solver)
add_subdirectory(src/uniterm)
add_subdirectory(src/utils)
add_subdirectory(src/video)

if(CUDA_FOUND)
    add_subdirectory(src/tensorrt)
    add_subdirectory(cuda)
endif()


# 添加目标文件
if(CUDA_FOUND)
    set(
        TARGETS_LIST
            dprm_attack
            dprm_kalman
            dprm_pointer
            dprm_solver
            dprm_delay
            dprm_print
            dprm_serial
            dprm_tf
            dprm_timer
            dprm_uniterm
            dprm_video
            dprm_tensorrt
            dprm_cudatools
    )
else()
    set(
        TARGETS_LIST
            dprm_attack
            dprm_kalman
            dprm_pointer
            dprm_solver
            dprm_delay
            dprm_print
            dprm_serial
            dprm_tf
            dprm_timer
            dprm_uniterm
            dprm_video
    )
endif()

# 安装
install(
    TARGETS
        ${TARGETS_LIST}
    EXPORT
        DPRMTarget
    RUNTIME DESTINATION
        bin
    LIBRARY DESTINATION
        lib
    ARCHIVE DESTINATION
        lib
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/dp_version.h
    DESTINATION
        include/dprm
)

if(CUDA_FOUND)
    install(
        FILES
            ${CMAKE_SOURCE_DIR}/cuda/include/cudatools.h
        DESTINATION
            include/dprm
    )
endif()

install(
    DIRECTORY
        ${CMAKE_SOURCE_DIR}/include/
    DESTINATION
        include/dprm
)

install(
    EXPORT
        DPRMTarget
    NAMESPACE
        dprm::
    DESTINATION
        lib/cmake/dprm
)

# 帮助自动生成cmakelist初始化代码
include(CMakePackageConfigHelpers)

if(CUDA_FOUND)
    configure_package_config_file(
        ${CMAKE_SOURCE_DIR}/cmake/DPRMConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/DPRMConfig.cmake
        INSTALL_DESTINATION
            lib/cmake/dprm
    )
else()
    configure_package_config_file(
        ${CMAKE_SOURCE_DIR}/cmake/DPRMConfig.cmake.in.nocuda
        ${CMAKE_CURRENT_BINARY_DIR}/DPRMConfig.cmake
        INSTALL_DESTINATION
            lib/cmake/dprm
    )
endif()


install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/DPRMConfig.cmake
    DESTINATION
        lib/cmake/dprm
)

# 卸载
if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY
    )

    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
    )
endif()
