#ifndef OPTIMIZER_SAMPLER_H
#define OPTIMIZER_SAMPLER_H

#include <better_flow/common.h>
#include <better_flow/event.h>
#include <better_flow/event_file.h>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>


class OptimizerLocal {
protected:
    LinearEventCloud *events;
    Event event_c;

    cv::Mat project_img;

    int scale;
    int metric_wsizex, metric_wsizey;

    int scale_img_x;
    int scale_img_y;

    // Gradient descent variables
    double nx, ny;
    double last_score, dscore;
    double dnx, dny, dn_th;

public:
    OptimizerLocal(LinearEventCloud *events_, Event &e_, int sc_, int wsz_) : 
        events(events_), event_c(e_), scale(sc_), metric_wsizex(sc_ * wsz_), metric_wsizey(sc_ * wsz_) {
        this->update_fields();        
    }

    OptimizerLocal(LinearEventCloud *events_, int sc_) : 
        events(events_), scale(sc_) {

        // Get the size of the window so all events fit in
        int x_min = this->events->x_min, y_min = this->events->y_min;
        int x_max = this->events->x_max, y_max = this->events->y_max;
       
        this->metric_wsizex = sc_ * (x_max - x_min);
        this->metric_wsizey = sc_ * (y_max - y_min);

        this->event_c = Event((x_max - x_min) / 2 + x_min, (y_max - y_min) / 2 + y_min, 0);
        this->update_fields();
    }

    int run ();
    int manual ();

    inline double get_nx() {return this->nx; }
    inline double get_ny() {return this->ny; }

private:
    double compute_new_nx(double nx_);
    double compute_new_ny(double ny_);

    inline double iteration_step (double nx_, double ny_);
    inline double iteration_step_debug (double nx_, double ny_);
    inline double get_event_score ();
    void update_fields ();

protected:
    auto begin() {return this->events->begin(); }
    auto end() {return this->events->end(); }
};


#endif // OPTIMIZER_SAMPLER_H
