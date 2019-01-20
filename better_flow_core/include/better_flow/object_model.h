#ifndef OBJECT_MODEL_H
#define OBJECT_MODEL_H

#include <better_flow/common.h>
#include <better_flow/event_file.h>


class ObjectModel {
public:
    double cx, cy, dx, dy, rot, div;
    uint cnt;

    double total_dx, total_dy, total_rot, total_div;

    ObjectModel () 
        : cx(0), cy(0), dx(0), dy(0), rot(0), div(0), cnt(0),
          total_dx(0), total_dy(0), total_rot(0), total_div(0) {}
    
    ObjectModel (cv::Mat time_img_)        
        : cx(0), cy(0), dx(0), dy(0), rot(0), div(0), cnt(0),
          total_dx(0), total_dy(0), total_rot(0), total_div(0) {
        this->update(time_img_);
    }

    ObjectModel (cv::Mat time_img_, double cx_, double cy_)
        : cx(0), cy(0), dx(0), dy(0), rot(0), div(0), cnt(0),
          total_dx(0), total_dy(0), total_rot(0), total_div(0) {
        this->update(time_img_, cx_, cy_);
    }

    void update (cv::Mat time_img_) {
        this->center_of_mass(time_img_);
        this->compute(time_img_);
    }

    void update (cv::Mat time_img_, double cx_, double cy_) {
        this->cx = cx_;
        this->cy = cy_;
        this->compute(time_img_);
    }

    void update (cv::Mat &time_img_, LinearEventCloud *events,
                 float scale, float x_shift, float y_shift, float p) {
        this->center_of_mass(time_img_);
        this->compute(time_img_, events, scale, x_shift, y_shift, p);
    }

    inline void update_accumulators (float d1, float d2, float d3, float d4) {
        this->total_rot += this->rot / d1;
        this->total_div += this->div / d2;
        this->total_dx += this->dx / d3;
        this->total_dy += this->dy / d4;
    }

    friend std::ostream &operator<< (std::ostream &output, const ObjectModel &M) { 
        output << "C: (" << M.cx << ", " << M.cy << "); " << std::endl
               << "\t Shift: (" << M.dx << ", " << M.dy << "); "
               << " total: (" << M.total_dx << ", " << M.total_dy << ");" << std::endl
               << "\t Rot: " << M.rot << " total: " << M.total_rot << std::endl
               << "\t Div: " << M.div << " total: " << M.total_div << std::endl
               << "\t cnt: " << M.cnt << std::endl;
        return output;            
    }
    
    void compute (cv::Mat &time_img);
    void compute (cv::Mat &time_img, LinearEventCloud *events,
                  float scale, float x_shift, float y_shift, float p);

    void center_of_mass (cv::Mat &time_img);

    static bool sobel_point(cv::Mat &img, int i, int j, float &dx, float &dy);
};


#endif // OBJECT_MODEL_H