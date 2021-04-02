#include <better_flow/common.h>
#include <better_flow/event.h>
#include <better_flow/event_file.h>
#include <better_flow/optimizer_global.h>
#include <better_flow/optimizer_sampler.h>
#include <better_flow/optimizer_rolling.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/videoio/videoio.hpp>

#include <queue>


std::string f2str(double v) {
    int base = int(v * 100);
    std::string ret = std::to_string(base / 100) + ".";
    ret += std::to_string(std::abs(base) % 100);
    return ret;
}

template <size_t MAX_SZ, sll SPAN> class DVS_flow {
public:
    // Buffer for incoming events (aka 'slice')
    CircularArray<Event, MAX_SZ, SPAN> ev_buffer;

protected:
    // Triggers for starting the processing
    //  on certain number of new events
    //  or when enough time passed
    ull on_ev_change, on_time_change;

    // Various time / event counters
    sll time_diff, event_diff; // Time passed / new event count

    // Timestamps of the last slice and the current slice
    ull last_slice_time, current_slice_time;

    // Last known model - starting point for the new minimizer
    ObjectModel last_model; 

    // Helper variables and data structures
    bool accumulate;
    std::vector<LinearEventCloudTemplate<Event>> accumulated;

    // Long term motion compenstation
    std::deque<std::pair<LinearEventCloud, ObjectModel>> motion_memory;

    // Set the mode for the minimizer (for debugging purpose)
    bool manual_mode;

    // Set maximum number of optimization steps (for low-latency applications)
    int max_iter;

    // Set the scale of the images (1 / 3 / ...) used in minimization
    int scale;

    // Specify if we shall write a video file with results
    bool generate_video;
    int video_fps;

    // Specify if we shall write a picture after each iteration 
    bool generate_pictures;
    std::string img_prefix;

    // Specify whether to use the estimate from the previous slice
    // as a starting point for minimization
    bool stm_disable;

    // Video writer object
    cv::VideoWriter *outputvideo;
    ull frame_count;

public:

    // Provide the triggers for event processing and optionally set the time 
    // stamp of the very first incoming event (so that the 'on_time_change'
    // trigger would not fire on the first event)
    DVS_flow (ull on_ev_change_, ull on_time_change_, ull start_time = 0) 
        : on_ev_change(on_ev_change_), on_time_change(on_time_change_),
          time_diff(0), event_diff(0), last_slice_time(start_time), 
          current_slice_time(start_time), accumulate(false), manual_mode(false),
          max_iter(-1), scale(3), generate_video(false), generate_pictures(false),
          stm_disable(false), outputvideo(NULL), frame_count(0) {}

    // Clear up the memory
    ~DVS_flow () {
        if (this->outputvideo != NULL) delete this->outputvideo;
    }

    // Add a new event to the buffer. The processing will start automatically
    // if one of the triggers is set, otherwise nothing happens. 
    // Returns true if the processing is done, false otherwise
    bool add_event (Event &ev);

    // Perform the computetion with the data currently available
    // This function is automatically called by 'add_event' when 
    // one of the triggers fire
    void recompute ();

    // Accumulate all the processed events in a separate buffer
    // for offline post processing
    void set_accumulate (bool val = true) {this->accumulate = val; }
    LinearEventCloudTemplate<Event> get_accumulated ();

    // The manual mode flag is being passed to the minimizer
    void set_manual_mode (bool val = true) {this->manual_mode = val; }

    // Set maximum number of optimization steps (for low-latency applications)
    void set_max_iter (int val = -1) {this->max_iter = val; }

    // Set the scale of the images (1 / 3 / ...) used in minimization
    void set_scale (int val = 3) {this->scale = val; }

    // Generate a video file with flow in the end
    void set_generate_video (bool val = true, std::string name = "out.avi", int framerate = 30) {
        int w = 3 * RES_Y * 3;
        int h = RES_X * 3;
        this->video_fps = framerate;
        this->generate_video = val;
        if (this->generate_video) {
        #if CV_MAJOR_VERSION == 2
            this->outputvideo = new cv::VideoWriter(name, CV_FOURCC('W','M','V','2'), 
                                                   framerate, cv::Size(w, h), true);
        #elif CV_MAJOR_VERSION >= 3
            this->outputvideo = new cv::VideoWriter(name, cv::VideoWriter::fourcc('W','M','V','2'), 
                                                   framerate, cv::Size(w, h), true);
        #endif
        }
    }

    // Save a picture after each iteration
    void set_generate_pictures (bool val = true, std::string img_prefix_ = "./") {
        this->generate_pictures = val;
        this->img_prefix = img_prefix_;
    }

    // Specify whether to use the estimate from the previous slice
    // as a starting point for minimization
    void set_stm_disable (bool val = true) {this->stm_disable = val; }

    // Get access to various useful numbers:
    sll get_buf_size () {
        return this->ev_buffer.size();
    }

    sll get_time_diff () {
        return this->time_diff;
    }

    sll get_buf_time_diff () {
        ull slice_start_time = 0;
        if (this->ev_buffer.size() == MAX_SZ) {
            slice_start_time = this->ev_buffer[MAX_SZ - 1].timestamp;
        }
        else {
            slice_start_time = (this->current_slice_time > SPAN) ? this->current_slice_time - SPAN : 0;
        }
        return this->current_slice_time - slice_start_time;
    }
};


template <size_t MAX_SZ, sll SPAN>
bool DVS_flow<MAX_SZ, SPAN>::add_event (Event &ev) {
    this->ev_buffer.push_back(ev);
    
    this->event_diff ++;
    this->current_slice_time = ev.timestamp;

    // Assuming that time only increases
    this->time_diff = this->current_slice_time - this->last_slice_time;

    if ((this->event_diff < (sll)this->on_ev_change) &&
        (this->time_diff  < (sll)this->on_time_change)) {
        return false;
    }
    
    // This will update the flow of events in the current buffer
    this->recompute();
    return true;
}


template <size_t MAX_SZ, sll SPAN>
void DVS_flow<MAX_SZ, SPAN>::recompute () {
    ull slice_start_time = 0;
    if (this->ev_buffer.size() == MAX_SZ) {
        //std::cout << "Buffer oveflow; executing premature computation\n";
        slice_start_time = this->ev_buffer[MAX_SZ - 1].timestamp;
    }
    else {
        slice_start_time = (this->current_slice_time > SPAN) ? this->current_slice_time - SPAN : 0;
    }

    // TODO: ugly loop
    LinearEventPtrs e_ptrs;
    for (auto &e : this->ev_buffer)
        e_ptrs.push_back(&e);

    // Thq queue of 'objects' to process; An object is a pair of events and an object model
    std::queue<std::pair<LinearEventPtrs, ObjectModel>> task_queue;
    task_queue.push(std::make_pair(e_ptrs, this->last_model));

    cv::Mat img_arrow;


    while (!task_queue.empty()) {
        // All computations need to be conducted in the 'local' 'slice' time
        // Run the optimizer 
        OptimizerRolling<decltype(e_ptrs)> optimizer;
        optimizer.set_cloud(&task_queue.front().first, this->scale);
        optimizer.set_time(slice_start_time);
        optimizer.set_maxiter(this->max_iter);

        // We can speed up the minimization process by first applying estimate
        // from the previous slice. It can increase performance if the direction
        // of motion did not cahnge.
        if (!this->stm_disable)
            optimizer.set_model(task_queue.front().second);
        
        if (this->manual_mode) optimizer.manual();
        else optimizer.run();

        this->last_model = optimizer.get_model();

        if (this->generate_video || this->generate_pictures) {
            img_arrow = optimizer.get_gradient_img_color();
        }

        task_queue.pop();
    }

    // We need to compute the actual u and v after minimizations are done
    for (auto &e : this->ev_buffer)
        e.compute_uv();


    // Now save the stuff in memory for long term motion compensation:
    LinearEventCloud cur_buf;
    for (auto &e : this->ev_buffer)
        cur_buf.push_back(e);
    this->motion_memory.push_back(std::make_pair(cur_buf, this->last_model));

    // Show the combined motion compensated picture:
    std::cout << "\n\n------------------------\n";
    for (auto &slice : this->motion_memory) {
        LinearEventCloud &cl = slice.first;
        ObjectModel &m = slice.second;
        std::cout << m << "\n";
        std::cout << cl.size() << "\t" << cl[0].timestamp << "\t" 
                  << cl[cl.size() - 1].timestamp << "\n";
    }


    // Output the result in a form of a picture or a video file
    if (this->generate_video || this->generate_pictures) {
        cv::Mat img_pr_f    = EventFile::projection_img(&this->ev_buffer, 3, false);
        cv::Mat img_color_f = EventFile::color_time_img(&this->ev_buffer, 3, false);
        cv::Mat img_pr_t    = EventFile::projection_img(&this->ev_buffer, 3, true);
        cv::Mat img_color_t = EventFile::color_time_img(&this->ev_buffer, 3, true);

        //cv::Mat img_arrow = EventFile::arrow_flow_img(&this->ev_buffer);
        //cv::Mat img_color = EventFile::color_flow_img(&this->ev_buffer);

        cv::cvtColor(img_pr_t, img_pr_t, cv::COLOR_GRAY2RGB);
        cv::cvtColor(img_pr_f, img_pr_f, cv::COLOR_GRAY2RGB);
        
        cv::resize(img_pr_t, img_pr_t, cv::Size(RES_Y * 3, RES_X * 3));
        cv::resize(img_pr_f, img_pr_f, cv::Size(RES_Y * 3, RES_X * 3));
        //cv::resize(img_arrow, img_arrow, cv::Size(RES_Y * 3, RES_X * 3));
        cv::resize(img_color_t, img_color_t, cv::Size(RES_Y * 3, RES_X * 3));
        cv::resize(img_color_f, img_color_f, cv::Size(RES_Y * 3, RES_X * 3));
  
        double slice_time_width = double(this->time_diff) / 1000000000.0;
        double speedup = double(this->on_time_change) / double(this->time_diff);

        cv::putText(img_pr_t, "timestamp: " + f2str(double(this->current_slice_time) / 1000000000.0), 
                    cv::Point(20, 40), cv::FONT_HERSHEY_DUPLEX, 0.6, 
                    cv::Scalar(255,255,255), 1, cv::LINE_AA, false);
        cv::putText(img_pr_t, "%realtime: " + f2str(speedup), 
                    cv::Point(20, 70), cv::FONT_HERSHEY_DUPLEX, 0.6, 
                    cv::Scalar(255,255,255), 1, cv::LINE_AA, false);
        cv::putText(img_pr_t, "Time diff (new): " + f2str(slice_time_width), 
                    cv::Point(20, 100), cv::FONT_HERSHEY_DUPLEX, 0.6, 
                    cv::Scalar(255,255,255), 1, cv::LINE_AA, false);
        cv::putText(img_pr_t, "Events: " + std::to_string(this->ev_buffer.size()), 
                    cv::Point(20, 130), cv::FONT_HERSHEY_DUPLEX, 0.6, 
                    cv::Scalar(255,255,255), 1, cv::LINE_AA, false);
        cv::putText(img_pr_t, "New events: " + std::to_string(this->event_diff), 
                    cv::Point(20, 160), cv::FONT_HERSHEY_DUPLEX, 0.6, 
                    cv::Scalar(255,255,255), 1, cv::LINE_AA, false);

        // Model
        cv::putText(img_pr_f, "Model:", 
                    cv::Point(20, RES_X * 3 - 160), cv::FONT_HERSHEY_DUPLEX, 0.6, 
                    cv::Scalar(255,255,255), 1, cv::LINE_AA, false);
        cv::putText(img_pr_f, "C: (" + f2str(this->last_model.cx) + ", " 
                    + f2str(this->last_model.cy) + ")",
                    cv::Point(20, RES_X * 3 - 130), cv::FONT_HERSHEY_DUPLEX, 0.6, 
                    cv::Scalar(255,255,255), 1, cv::LINE_AA, false);
        cv::putText(img_pr_f, "Shift: (" + f2str(this->last_model.dx) + ", " 
                    + f2str(this->last_model.dy) + "); total: ("
                    + f2str(this->last_model.total_dx) + ", "
                    + f2str(this->last_model.total_dy) + ")", 
                    cv::Point(20, RES_X * 3 - 100), cv::FONT_HERSHEY_DUPLEX, 0.6, 
                    cv::Scalar(255,255,255), 1, cv::LINE_AA, false);
        cv::putText(img_pr_f, "Rot: " + f2str(this->last_model.rot) + " total: "
                    + f2str(this->last_model.total_rot), 
                    cv::Point(20, RES_X * 3 - 70), cv::FONT_HERSHEY_DUPLEX, 0.6, 
                    cv::Scalar(255,255,255), 1, cv::LINE_AA, false);
        cv::putText(img_pr_f, "Div: " + f2str(this->last_model.div) + " total: "
                    + f2str(this->last_model.total_div), 
                    cv::Point(20, RES_X * 3 - 40), cv::FONT_HERSHEY_DUPLEX, 0.6, 
                    cv::Scalar(255,255,255), 1, cv::LINE_AA, false);

        cv::Mat concat;
        cv::hconcat(img_pr_t, img_color_t, concat);

        cv::Mat bottom;
        cv::hconcat(img_pr_f, img_color_f, bottom);

        cv::vconcat(concat, bottom, concat);

        if (this->generate_pictures) {
            cv::imwrite(this->img_prefix + "/frame_" + std::to_string(this->frame_count) + ".jpg", concat);        
            this->frame_count ++;
        }

        if (this->generate_video) {
            if (!outputvideo->isOpened()){
                std::cout  << "Could not open the output video for write" << std::endl;
            }
            (*this->outputvideo) << concat;
        }
    }

    this->event_diff = 0;
    this->last_slice_time = this->current_slice_time;

    // Save the results
    if (this->accumulate) {
        LinearEventCloudTemplate<Event> cur_buf;
        for (long int i = (long int)this->ev_buffer.size() - 1; i >= 0; i--)
            cur_buf.push_back(this->ev_buffer[i]);
        this->accumulated.push_back(cur_buf);
    }
}


template <size_t MAX_SZ, sll SPAN>
LinearEventCloudTemplate<Event> DVS_flow<MAX_SZ, SPAN>::get_accumulated () {
    LinearEventCloudTemplate<Event> ret;
    std::cout << "Aggregating events into one cloud...\n";

    for (ull i = 0; i < this->accumulated.size(); ++i) {
        std::cout << "\tBuffer: " << i << "\n";

        auto &buf = this->accumulated[i];
        for (auto &e : buf) {
            if (e.t == -1) continue;
            //if (e.noise) continue;
            
            Event ev = e;
            float avg_cnt = 1;

            for (ull j = i + 1; j < this->accumulated.size(); ++j) {
                auto &buf_next = this->accumulated[j];
                for (auto &e_ : buf_next) {

                    if (e_ - e > 0) break;
                    if (e_.t == -1) continue;
                    if (e != e_) continue;
                    e_.t = -1;
                     
                    //ev.best_u += e_.best_u;
                    //ev.best_v += e_.best_v;
                    //avg_cnt += 1;
                }
            }

            ev.best_u /= avg_cnt;
            ev.best_v /= avg_cnt;
            ret.push_back(ev);
        }
    }

    std::cout << "FInal buffer contains " << ret.size() << " events." << std::endl;
    return ret;
}
