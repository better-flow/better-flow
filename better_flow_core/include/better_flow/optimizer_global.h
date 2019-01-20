#ifndef OPTIMIZER_GLOBAL_H
#define OPTIMIZER_GLOBAL_H

#include <better_flow/common.h>
#include <better_flow/event.h>
#include <better_flow/event_file.h>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>


class OptimizerGlobal {
protected:
    LinearEventCloud *events;

    cv::Mat project_img;
    cv::Mat current_scores;

    int scale;
    int metric_wsize;

    int scale_img_x;
    int scale_img_y;
    int scale_bordered_img_x;
    int scale_bordered_img_y;

public:
    OptimizerGlobal(LinearEventCloud *events_) : 
        events(events_), scale(5), metric_wsize(21) {
        this->update_fields ();        
    }

    OptimizerGlobal(LinearEventCloud *events_, int sc_) : 
        events(events_), scale(sc_), metric_wsize(5 * sc_) {
        this->update_fields ();
    }

    OptimizerGlobal(LinearEventCloud *events_, int sc_, int wsz_) : 
        events(events_), scale(sc_), metric_wsize(wsz_) {
        this->update_fields ();
    }

    void project_all (double nx_, double ny_, double nz_ = NZ);
    void compute_flow_bruteforce ();
    int manual (double nx_ = 0, double ny_ = 0);

private:
    inline double get_event_score (int x, int y);
    void update_fields ();

protected:
    auto begin() {return this->events->begin(); }
    auto end() {return this->events->end(); }
};


#endif // OPTIMIZER_GLOBAL_H
