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

target_sources(
    dprm_video
        PRIVATE
        ${CMAKE_SOURCE_DIR}/src/video/uvc.cpp
        ${CMAKE_SOURCE_DIR}/src/video/tools.cpp
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
        $<IF:$<BOOL:${HAVE_GXIAPI}>,gxiapi,>
)
