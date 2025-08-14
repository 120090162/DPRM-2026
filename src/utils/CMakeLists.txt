add_library(
    dprm_delay
        SHARED
)
target_sources(
    dprm_delay
        PRIVATE
        ${CMAKE_SOURCE_DIR}/src/utils/delay.cpp
)
target_include_directories(
    dprm_delay
        PRIVATE
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include/dprm>
)



add_library(
    dprm_print
        SHARED
)
target_sources(
    dprm_print
        PRIVATE
        ${CMAKE_SOURCE_DIR}/src/utils/print.cpp
)
target_include_directories(
    dprm_print
        PRIVATE
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include/dprm>
)




add_library(
    dprm_serial
        SHARED
)
target_sources(
    dprm_serial
        PRIVATE
        ${CMAKE_SOURCE_DIR}/src/utils/serial.cpp
)
target_include_directories(
    dprm_serial
        PRIVATE
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include/dprm>
)




add_library(
    dprm_tf
        SHARED
)
target_sources(
    dprm_tf
        PRIVATE
        ${CMAKE_SOURCE_DIR}/src/utils/tf_rotate.cpp
        ${CMAKE_SOURCE_DIR}/src/utils/tf_trans.cpp
        ${CMAKE_SOURCE_DIR}/src/utils/tf_tools.cpp
)
target_include_directories(
    dprm_tf
        PRIVATE
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include/dprm>
)




add_library(
    dprm_timer
        SHARED
)
target_sources(
    dprm_timer
        PRIVATE
        ${CMAKE_SOURCE_DIR}/src/utils/timer.cpp
)
target_include_directories(
    dprm_timer
        PRIVATE
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include/dprm>
)
