cmake_minimum_required(VERSION 2.8.3)
project(better_flow)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -g")
set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)


set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -Werror")

find_package(TBB REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenCL)

###########
## Build ##
###########

if (NOT OPENCL_FOUND)
    set(OpenCL_LIBRARY "")
    set(OpenCL_INCLUDE_DIRS "")
endif (NOT OPENCL_FOUND)

add_definitions(-DOPENCL_ENABLED=${OPENCL_FOUND})

include_directories(
    # Better Flow include files
    ${CMAKE_CURRENT_SOURCE_DIR}/include

    # Intel Thread Building Blocks
    ${TBB_INCLUDE_DIR}

    # GPU acceleration
    ${OpenCL_INCLUDE_DIRS}
)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# core dev binary
add_executable(bf_motion_compensator

    # Header files
    include/better_flow/common.h
    include/better_flow/event_file.h
    include/better_flow/event.h
    include/better_flow/clustering.h
    include/better_flow/object_model.h
    include/better_flow/optimizer_global.h
    include/better_flow/optimizer_sampler.h
    include/better_flow/optimizer_rolling.h
    include/better_flow/opencl_driver.h

    # Source code files
    src/bf_motion_compensator.cpp
    src/event_file.cpp
    src/clustering.cpp
    src/object_model.cpp
    src/optimizer_global.cpp
    src/optimizer_sampler.cpp
    src/optimizer_rolling.cpp
    src/opencl_driver.cpp
)

target_link_libraries(bf_motion_compensator
    ${OpenCV_LIBS}
    ${TBB_IMPORTED_TARGETS} tbb ${OpenCL_LIBRARY}
)

# viewer for event file manipulation
add_executable(bf_viewer

    # Source code files
    src/bf_viewer.cpp
)

target_link_libraries(bf_viewer
    ${OpenCV_LIBS}
)
