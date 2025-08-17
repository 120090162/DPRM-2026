add_library(
    dprm_video
        SHARED
)

# 查找 gxiapi 动态链接库
find_library(GXIAPI_LIB
    NAMES gxiapi
)

# 检查库是否找到
if(GXIAPI_LIB)
    message(STATUS "Found DaHeng: ${GXIAPI_LIB}")
    set(HAVE_GXIAPI TRUE)
else()
    message(STATUS "Could NOT found DaHeng libgxiapi.so, disable DaHeng camera support.")
    set(HAVE_GXIAPI FALSE)
endif()

# 检查MVS驱动
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  set(HIKSDK_LIB_DIR "${CMAKE_SOURCE_DIR}/lib/HikSDK/amd64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(HIKSDK_LIB_DIR "${CMAKE_SOURCE_DIR}/lib/HikSDK/arm64")
else()
  message(FATAL_ERROR "Unsupported host system architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}!")
endif()

# 将库目录添加到链接路径 - 使用绝对路径
target_link_options(dprm_video PUBLIC "-L${HIKSDK_LIB_DIR}")
# 安装库文件
install(
    DIRECTORY "${HIKSDK_LIB_DIR}/"
    DESTINATION lib
)

target_sources(
    dprm_video
        PRIVATE
        ${CMAKE_SOURCE_DIR}/src/video/uvc.cpp
        ${CMAKE_SOURCE_DIR}/src/video/tools.cpp
        ${CMAKE_SOURCE_DIR}/src/video/hikang.cpp
        $<IF:$<BOOL:${HAVE_GXIAPI}>,${CMAKE_SOURCE_DIR}/src/video/daheng.cpp,>
)

target_include_directories(
    dprm_video
        PRIVATE
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include/dprm>
)

target_link_libraries(
    dprm_video
        PRIVATE
        FormatConversion
        MediaProcess
        MvCameraControl
        MVRender
        MvUsb3vTL
        $<IF:$<BOOL:${HAVE_GXIAPI}>,gxiapi,>
)
