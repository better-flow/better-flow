Better Flow
=============

A motion-compensation pipeline for the DVS/DAVIS event-based cameras.

## Disclaimer:
**NOTE** the detection and segmentation pipeline has not been released yet. This code only includes motion compensation using time image.

1. The code was tested on Ubuntu 16.04 and 17.04 and consists of multiple tools for visuzlization, evaluation and
processing the event camera data.

2. The code supports integration with ROS ([ROS Kinetic](http://wiki.ros.org/kinetic)) but most code can be run as a separate binary with no need for ROS installation.
Only ROS frontend supports reading data from the camera. Some useful ROS links:
  - [ROS Kinetic installation](http://wiki.ros.org/kinetic/Installation/Ubuntu "Read this to install ROS on your system")
  - [ROS Tutorials](http://wiki.ros.org/ROS/Tutorials "This is a set of brief ROS tutorials")


## Setup with ROS frontend (may be broken, see standalone instructions below)
### First time setup:
1. Make sure ROS Kinetic is [installed](http://wiki.ros.org/kinetic/Installation/Ubuntu) on you [Ubuntu 16.04](http://releases.ubuntu.com/16.04/) system. Other configurations are possible but not tested.
2. Download the *cognifli* code (see [project page](https://github.com/ncos/cognifli) for more details):
```
cd ~/
git clone https://github.com/ncos/cognifli
```
3. Run the *INSTALL.py* configuration tool to set up your catkin workspace:
```
cd ~/cognifli/contrib
./INSTALL.py
```
4. Download DVS [camera drivers](https://github.com/uzh-rpg/rpg_dvs_ros):
```
cd ~/cognifli/src
git clone https://github.com/catkin/catkin_simple.git
git clone https://github.com/uzh-rpg/rpg_dvs_ros.git
```

5. Download Better Flow [source code](https://github.com/better-flow/better-flow):
```
cd ~/cognifli/src
git clone https://github.com/better-flow/better-flow
cd better_flow
git submodule init
git submodule update
```

6. Build the project:
```
cd ~/cognifli
catkin_make
```

**Note!** you might get compilation errors form the *dvs_calibration* package, you can just remove it:
```
rm -rf ~/cognifli/src/rpg_dvs_ros/dvs_calibration
```

7. Install the udev rule (in a new terminal window):
```
roscd libcaer_catkin
sudo ./install.sh
```

### Running the code:
1. To launch the camera driver, type:
```
roslaunch dvs_renderer dvs_mono.launch # For DVS sensor
roslaunch dvs_renderer davis_mono.launch # For DAVIS sensor
```

2. To record a *.bag* file, type (do not forget to run *roscore* in a separate terminal!):
```
rosbag record <topic_1> <topic_2> ... <topic_n> -o <output_file_name>
```

Useful topic names are:
  - */dvs/imu*
  - */dvs/events* 
  - */dvs/image_raw*
  - */dvs_renderer*


3. To play back the *.bag* file (do not forget to run *roscore* in a separate terminal!):
```
rosbag play <file_name>
```
Also check manuals [here](http://wiki.ros.org/rqt_bag) and [here](http://wiki.ros.org/rosbag/Commandline)

4. To convert a *.bag* file in *.txt* format (for future offline processing), use Python scripts at:
```
~/cognifli/src/rpg_dvs_ros/dvs_file_writer/scripts
```

5. To run the code:
  - The visualizer:
    ```
    roslaunch better_flow better_visualizer.launch
    ```

  - The visualizer with motion-compensation:
    ```
    roslaunch better_flow better_proc_visualizer.launch
    ```

## ROS-independent setup
### First time setup:
1. Install dependencies
```
sudo apt install libtbb-dev
```

Build and install OpenCV from source with QT support turned on. QT support is not turned on in the version from the Ubuntu repositories. E.g. build OpenCV with
```
cmake -DWITH_QT=ON ../.
```
Instructions for building and installing OpenCV from source can be found here: `https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html`


2. Download Better Flow [source code](https://github.com/better-flow/better-flow) and build the project:
```
git clone https://github.com/better-flow/better-flow
cd better_flow/better_flow_core
cmake .
make -j8
```

### Running the code:
1. The main binary is called *bf_motion_compensator*. Run 
```
./bf_motion_compensator -h
```
to get a list of options. This binary reads the event file in the *.txt* format. Use *'-i'* option to perform motion
compensation step-by-step.


2. The *bf_viewer* is designed to view event file recordings.



## Code organization
  - *better_flow/ros_nodes_src* - this folder contains ROS frontend code, it reads the data from the camera and publishes
  visualization data to show in Rviz
  - *better_flow/better_flow_core/include/better_flow* - most of the code is located in header files (many classes are templates)
    - *dvs_flow.h* contains the main class which aggregates all the low-level primitives of the motion-compensation
    pipeline. It manages the width of the time slice and provides a lot of other high level functionality.
    - *optimizer.h* contains the code for the optimization algorithm (currently - gradient descent)
    - *accel_lib.h* contains all the low-level code - generation of event-count and time images, sobel filter and so on
    - *event_file.h* consists of visualization code (display color-coded flow, etc.) the visuzlization is done with
    OpenCV
    - *event.h* contains the class which represents a single event and all metadata associated with it.
    - *datastructures.h* contains the implementation of the circular array used to store the time slice
  - *better_flow/better_flow_core/src* - the implementation of the interfaces provided in corresponding header files
    - *bf_motion_compensator.cpp* is the *main* file for the motion-compensation binary, it is responsible for parsing
    cli arguments, reading the input event file and managing the *dvs_flow* class.
    - *bf_viewer.cpp* is the *main* file for the event file viewer
