cmake_minimum_required(VERSION 2.8)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_BUILD_TYPE Release)

if ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
    set(CMAKE_C_COMPILER gcc)
    set(CMAKE_CXX_COMPILER g++)
else()
    set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(CURL REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}/rknn_api/include
)
set(link_libs ${CMAKE_SOURCE_DIR}/rknn_api/lib64/librknn_api.so pthread opencv_core opencv_highgui opencv_imgcodecs opencv_imgproc opencv_videoio)

add_executable(yolov3-320 yolov3_320.cpp  src/v4l2/v4l2.cpp  src/screen/screen.cpp  src/config.cpp)
target_link_libraries(yolov3-320 ${link_libs})

add_executable(yolov3-416 yolov3_416.cpp  src/v4l2/v4l2.cpp  src/screen/screen.cpp  src/config.cpp)
target_link_libraries(yolov3-416 ${link_libs})

add_executable(yolov3-608 yolov3_608.cpp  src/v4l2/v4l2.cpp  src/screen/screen.cpp  src/config.cpp)
target_link_libraries(yolov3-608 ${link_libs})

add_executable(yolov3-tiny yolov3_tiny.cpp  src/v4l2/v4l2.cpp  src/screen/screen.cpp  src/config.cpp)
target_link_libraries(yolov3-tiny ${link_libs})
