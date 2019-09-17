#include <vector>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <iomanip>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common_headers.h>
#include <pcl/filters/passthrough.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/image_encodings.h>
#include <message_filters/subscriber.h>

#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>

#include <prophesee_event_msgs/PropheseeEvent.h>
#include <prophesee_event_msgs/PropheseeEventBuffer.h>

#include <samsung_ros_driver/SamsungEvent.h>
#include <samsung_ros_driver/SamsungEventBuffer.h>

#include <better_flow/common.h>
#include <better_flow/event.h>
#include <better_flow/dvs_flow.h>


#define EVENT_WIDTH 5000000
#define TIME_WIDTH 0.5

#define EVENT_WIDTH_PROCESS 30000
#define TIME_WIDTH_PROCESS 0.07


// Node launch parameters (set in main)
float refresh_rate;
std::string input_event_topic_dvs;
std::string input_event_topic_prophesee;
std::string input_event_topic_samsung;
std::string input_image_topic;
std::string output_image_topic;
std::string output_pointcloud_topic;
std::string output_image_s0_topic;
std::string output_image_s1_topic;
std::string output_image_s2_topic;
std::string input_file;
bool process_data;

// Main class
template <size_t MAX_SZ, sll SPAN> class EventVisualizer {
protected:
    ros::NodeHandle n_;
    image_transport::ImageTransport it_;

    // Publishers/Subscribers
    ros::Subscriber event_sub_dvs, event_sub_prophesee, event_sub_samsung;
    image_transport::Subscriber image_sub;
    ros::Publisher cloud_pub;
    image_transport::Publisher image_pub;
    image_transport::Publisher suppl_image_pub_0,
                               suppl_image_pub_1, suppl_image_pub_2;

    // Buffer for incoming events (aka 'slice')
    CircularArray<Event, MAX_SZ, SPAN> ev_buffer;

    // Helper variables
    ull event_cnt;
    ros::Time start_system_time;
    ull first_event_timestamp;
    bool first_event_received;

    // Triggers for starting the processing
    //  on certain number of new events
    //  or when enough time passed
    ull on_ev_change, on_time_change;

    // Various time / event counters
    sll time_diff, event_diff; // Time passed / new event count

    // Timestamps of the last slice and the current slice
    ull last_slice_time, current_slice_time;

    // Flow estimator / minimizer
    DVS_flow<EVENT_WIDTH_PROCESS, FROM_SEC(TIME_WIDTH_PROCESS)> *estimator;

public:
    EventVisualizer (ros::NodeHandle n, ull on_ev_change_, ull on_time_change_) :
        n_(n), it_(n), event_cnt(0), start_system_time(ros::Time::now()),
        first_event_timestamp(0), first_event_received(false),
        on_ev_change(on_ev_change_), on_time_change(on_time_change_),
        time_diff(0), event_diff(0), last_slice_time(0), current_slice_time(0),
        estimator(NULL) {
        this->event_sub_dvs = this->n_.subscribe(input_event_topic_dvs, 0, &EventVisualizer::event_cb_dvs, this);
        this->event_sub_prophesee = this->n_.subscribe(input_event_topic_prophesee, 0,
                                                       &EventVisualizer::event_cb_prophesee, this);
        this->event_sub_samsung   = this->n_.subscribe(input_event_topic_samsung, 0,
                                                       &EventVisualizer::event_cb_samsung, this);
        this->image_sub = this->it_.subscribe(input_image_topic, 1, &EventVisualizer::image_cb, this);
        this->cloud_pub = n_.advertise<pcl::PointCloud<pcl::PointXYZRGB> > (output_pointcloud_topic, 1);
        this->image_pub = this->it_.advertise(output_image_topic, 1);

        if (process_data) {
            this->estimator = new DVS_flow<EVENT_WIDTH_PROCESS, FROM_SEC(TIME_WIDTH_PROCESS)>
                                          (20000000, LLONG_MAX);
            this->estimator->set_stm_disable(false);
            this->estimator->set_scale(1);
            this->estimator->set_max_iter(10);

            this->suppl_image_pub_0 = this->it_.advertise(output_image_s0_topic, 1);
            this->suppl_image_pub_1 = this->it_.advertise(output_image_s1_topic, 1);
            this->suppl_image_pub_2 = this->it_.advertise(output_image_s2_topic, 1);
        }
    }

    ~EventVisualizer() {
        if (this->estimator != NULL) delete this->estimator;
    }

    // Callbacks
    void event_cb_dvs(const dvs_msgs::EventArray::ConstPtr& msg) {
        if (!this->first_event_received && msg->events.size() != 0) {
            this->first_event_received = true;
            this->reset_lag_timers(msg->events[0].ts.toNSec());
        }

        for (uint i = 0; i < msg->events.size(); ++i) {
            ull time = msg->events[i].ts.toNSec();
            Event e(msg->events[i].x, msg->events[i].y, time);
            this->add_event(e);
            this->event_cnt++;
        }
    }

    void event_cb_prophesee(const prophesee_event_msgs::PropheseeEventBuffer::ConstPtr& msg) {
        if (!this->first_event_received && msg->events.size() != 0) {
            this->first_event_received = true;
            this->reset_lag_timers(msg->events[0].t * 1000);
        }

        for (uint i = 0; i < msg->events.size(); ++i) {
            ull time = msg->events[i].t * 1000;
            Event e(msg->events[i].x, msg->events[i].y, time);
            this->add_event(e);
            this->event_cnt++;
        }
    }

    void event_cb_samsung(const samsung_ros_driver::SamsungEventBuffer::ConstPtr& msg) {
        if (!this->first_event_received && msg->events.size() != 0) {
            this->first_event_received = true;
            this->reset_lag_timers(msg->events[0].t * 1000);
        }

        for (uint i = 0; i < msg->events.size(); ++i) {
            ull time = msg->events[i].t * 1000;
            Event e(msg->events[i].x, msg->events[i].y, time);
            this->add_event(e);
            this->event_cnt++;
        }
    }

    void image_cb(const sensor_msgs::ImageConstPtr& msg) {
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        this->image_pub.publish(cv_ptr->toImageMsg());
    }

    // Reset the timers which show algorithm lag
    void reset_lag_timers(ull current_time) {
        this->first_event_timestamp = current_time;
        this->start_system_time = ros::Time::now();
    }

    // Add a new event to the buffer. The processing will start automatically
    // if one of the triggers is set, otherwise nothing happens. 
    // Returns true if the processing is done, false otherwise
    bool add_event (Event &ev);

    // Visualize the data currently available
    // This function is automatically called by 'add_event' 
    void visualize ();

    // Visualize data coming from minimizer
    void visualize_minimizer ();
};


template <size_t MAX_SZ, sll SPAN>
bool EventVisualizer<MAX_SZ, SPAN>::add_event (Event &ev) {
    this->ev_buffer.push_back(ev);

    // Psh the event to the estimator as well (if it exists)
    if (this->estimator != NULL) estimator->add_event(ev);

    this->event_diff ++;
    this->current_slice_time = ev.timestamp;

    // Assuming that time only increases
    this->time_diff = this->current_slice_time - this->last_slice_time;

    if ((this->event_diff < (sll)this->on_ev_change) &&
        (this->time_diff  < (sll)this->on_time_change)) {
        return false;
    }

    // Measure lag
    const double time_events = double(this->current_slice_time - 
                                      this->first_event_timestamp) / 1000000000.0;
    const double time_system = (ros::Time::now() - this->start_system_time).toSec();
    const double time_lag = time_system - time_events;

    // Sometimes DVS timestamp values jump
    if (time_lag < -100000)
        this->reset_lag_timers(this->current_slice_time);

    std::cout << "Real time: " << time_system << "s.\t"
              << "Event time: " << time_events << "s.\t"
              << "Lag: ";
    if (fabs(time_lag) < 0.1)
        std::cout << _blue(time_lag);
    if (time_lag >  0.1)
        std::cout << _red(time_lag);
    if (time_lag < -0.1)
        std::cout << _green(time_lag);
    std::cout << "\n";

    // This will update the flow of events in the current buffer
    this->visualize();

    // This will generate and send the minimizer debug info
    this->visualize_minimizer();

    // Update the triggers
    this->event_diff = 0;
    this->last_slice_time = this->current_slice_time;
    return true;
}


template <size_t MAX_SZ, sll SPAN>
void EventVisualizer<MAX_SZ, SPAN>::visualize () {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    cloud->header.frame_id = "/base_link";

    sll t0 = this->ev_buffer[this->ev_buffer.size() - 1].timestamp;
    int color = 0xFF80DAEB; // Color: AA:RR:GG:BB

    const ull target_cloud_size = 200000; // Rviz cannot render more
    const int rate = this->ev_buffer.size() / target_cloud_size;

    ull i = 0;
    for (auto &e : this->ev_buffer) {
        ++i;

        if ((rate > 1) && (i % rate != 0))
            continue;

        pcl::PointXYZRGB p;
        p.rgba = color;

        p.x = (float)(RES_Y - e.fr_x) / (RES_X / 180 * 200);
        p.y = (float)e.fr_y / (RES_X / 180 * 200);
        p.z = double(e.timestamp - t0) / 1000000000.0;
        cloud->push_back(p);
    }

    this->cloud_pub.publish(cloud);
}


template <size_t MAX_SZ, sll SPAN>
void EventVisualizer<MAX_SZ, SPAN>::visualize_minimizer () {
    if (this->estimator == NULL) return;
    this->estimator->recompute(); // Run the minimizer

    cv::Mat image0;
    cv::transpose(EventFile::projection_img(&this->estimator->ev_buffer, 1), image0);
    sensor_msgs::ImagePtr msg0 = cv_bridge::CvImage(std_msgs::Header(), "mono8", image0).toImageMsg();
    this->suppl_image_pub_0.publish(msg0);

    cv::Mat image1;
    cv::transpose(EventFile::color_flow_img(&this->estimator->ev_buffer), image1);
    sensor_msgs::ImagePtr msg1 = cv_bridge::CvImage(std_msgs::Header(), "rgb8", image1).toImageMsg();

    cv::Mat image2;
    cv::transpose(EventFile::projection_img_unopt(&this->estimator->ev_buffer, 1), image2);
    sensor_msgs::ImagePtr msg2 = cv_bridge::CvImage(std_msgs::Header(), "mono8", image2).toImageMsg();

    this->suppl_image_pub_1.publish(msg1);
    this->suppl_image_pub_2.publish(msg2);
}


int main(int argc, char** argv) {
    ros::init(argc, argv, "better_event_visualizer");
    ros::NodeHandle nh;

    // topic names
    if (!nh.getParam("better_flow/input_event_topic_dvs", input_event_topic_dvs)) input_event_topic_dvs = "/dvs/events";
    if (!nh.getParam("better_flow/input_event_topic_prophesee", input_event_topic_prophesee))
        input_event_topic_prophesee = "/prophesee/camera/cd_events_buffer";
    if (!nh.getParam("better_flow/input_event_topic_samsung", input_event_topic_samsung))
        input_event_topic_samsung = "/samsung/camera/events";
    if (!nh.getParam("better_flow/input_image_topic", input_image_topic)) input_image_topic = "/dvs/image_raw";

    if (!nh.getParam("better_flow/output_image_topic", output_image_topic)) output_image_topic = "/bf/image";
    if (!nh.getParam("better_flow/output_pointcloud_topic", output_pointcloud_topic)) output_pointcloud_topic = "/bf/events";

    if (!nh.getParam("better_flow/output_image_s0_topic", output_image_s0_topic)) output_image_s0_topic = "/bf/additional/image0";
    if (!nh.getParam("better_flow/output_image_s1_topic", output_image_s1_topic)) output_image_s1_topic = "/bf/additional/image1";
    if (!nh.getParam("better_flow/output_image_s2_topic", output_image_s2_topic)) output_image_s2_topic = "/bf/additional/image2";

    // Refresh parameters
    if (!nh.getParam("better_flow/refresh_rate", refresh_rate)) refresh_rate = 15;
    if (!nh.getParam("better_flow/process_data", process_data)) process_data = false;

    // Read from text file
    if (!nh.getParam("better_flow/input_file", input_file)) input_file = "";

    // Compute refresh parameters
    float time_refresh = 1.0 / refresh_rate;
    ull event_refresh = LLONG_MAX;

    // Main class
    EventVisualizer<EVENT_WIDTH, FROM_SEC(TIME_WIDTH)> visualizer(nh, event_refresh, FROM_SEC(time_refresh));

    // ROS loop
    if (input_file == "") {
        ros::spin();
        return 0;
    }

    std::cout << "Reading from file... (" << input_file << ")" << std::endl << std::flush;
    std::cout << "The live camera will resume after file read is complete!" << std::endl << std::flush;

    // Read from event file
    std::ifstream event_file(input_file, std::ifstream::in);

    ull i = 0;
    double t = 0;
    uint x = 0, y = 0;
    bool p = false;

    double t_0 = 0; // the earliest timestamp in the file
    
    if (ros::ok() && event_file >> t_0 >> x >> y >> p) {
        ++i;
        Event e(y, x, FROM_SEC(0));
        visualizer.add_event(e);
    }

    while (ros::ok() && event_file >> t >> x >> y >> p) {
        t -= t_0;

        ++i;
        Event e(y, x, FROM_SEC(t));
        visualizer.add_event(e);
    }

    event_file.close();
    std::cout << "Read and processed " << i << " events" << std::endl << std::flush;
        
    ros::spin();
    return 0;
};
