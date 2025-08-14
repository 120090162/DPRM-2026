cmake_minimum_required(VERSION 3.15)
project(DPRM-Test)

set(CMAKE_CXX_STANDARD 11)
# add_compile_options(-g -O0 -w -fno-omit-frame-pointer -Wno-notes)

add_compile_options(-O3 -w)

# Cuda
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

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

add_executable(uvc_lib uvc_lib.cpp)
target_link_libraries(uvc_lib
    ${DPRM_LIBS}
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
)
