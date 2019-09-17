cmake_minimum_required(VERSION 2.8.3)
project(better_flow)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_BUILD_TYPE Release)

#set(CMAKE_BUILD_TYPE Debug)
add_definitions(-std=c++14)


add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/better_flow_core/ext
    ${CMAKE_CURRENT_SOURCE_DIR}/better_flow_core/ext_build
)


find_package(catkin REQUIRED COMPONENTS
				cv_bridge
				geometry_msgs
				image_transport
				message_generation
				pcl_conversions
				pcl_ros
				roscpp
				sensor_msgs
				visualization_msgs
				tf
                tf_conversions
				laser_geometry
				std_msgs
                dvs_msgs
                prophesee_event_msgs)

find_package(OpenCV REQUIRED)


catkin_package(CATKIN_DEPENDS message_runtime sensor_msgs std_msgs geometry_msgs dvs_msgs prophesee_event_msgs samsung_ros_driver)

###########
## Build ##
###########


include_directories(.
                    include
                    better_flow_core/include
                    combiner/include

                    # Intel Thread Building Blocks
                    ${TBB_INCLUDE_DIR}

                    ${catkin_INCLUDE_DIRS}
	                ${PCL_INCLUDE_DIRS})

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


add_executable(bf_visualizer_nd

    # Header files
    better_flow_core/include/better_flow/common.h
    better_flow_core/include/better_flow/event_file.h
    better_flow_core/include/better_flow/event.h
    better_flow_core/include/better_flow/clustering.h
    better_flow_core/include/better_flow/object_model.h
    better_flow_core/include/better_flow/optimizer_global.h
    better_flow_core/include/better_flow/optimizer_sampler.h
    better_flow_core/include/better_flow/optimizer_rolling.h
    better_flow_core/include/better_flow/opencl_driver.h

    # Source code files
    ros_nodes_src/bf_visualizer.cpp
    better_flow_core/src/event_file.cpp
    better_flow_core/src/clustering.cpp
    better_flow_core/src/object_model.cpp
    better_flow_core/src/optimizer_global.cpp
    better_flow_core/src/optimizer_sampler.cpp
    better_flow_core/src/optimizer_rolling.cpp
    better_flow_core/src/opencl_driver.cpp
)

target_link_libraries(bf_visualizer_nd
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${TBB_IMPORTED_TARGETS} tbb
    ${PCL_LIBRARIES}
)
