#ifndef OPTIMIZER_ROLLING_H
#define OPTIMIZER_ROLLING_H

#include <better_flow/common.h>
#include <better_flow/optimizer_global.h>
#include <better_flow/optimizer_sampler.h>
#include <better_flow/event.h>
#include <better_flow/event_file.h>
#include <better_flow/object_model.h>
#include <better_flow/accel_lib.h>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>



template <class T> class OptimizerRolling {
protected:
    AccelLib accel;

    T *events;

    int scale;
    int metric_wsizex, metric_wsizey;
    int max_itercount;

    int scale_img_x;
    int scale_img_y;

    double x_shift, y_shift;
    int x_min, y_min, x_max, y_max;

    ull current_time;

    ObjectModel model;
    float x_divider, y_divider, rot_divider, div_divider;
    bool manual_;

    cv::Mat time_img;
    cv::Mat sharpness_img;

public:
    OptimizerRolling() : events(NULL), scale(0), metric_wsizex(0), metric_wsizey(0), max_itercount(-1),
        scale_img_x(0), scale_img_y(0), x_shift(0), y_shift(0), x_min(0), y_min(0), 
        x_max(0), y_max(0), current_time(0), x_divider(1), y_divider(1), rot_divider(10000), div_divider(10000),
        manual_(false) {}

    int run () {
        if ((this->scale_img_x < this->scale * RES_X / 15) && (this->scale_img_y < this->scale * RES_Y / 15)) {
            if (VERBOSE) std::cout << "Window size is too small; (" << this->metric_wsizex
                                   << ", " << this->metric_wsizey << "). Skipping...\n";
            for (auto &e : *this->events)
                e.noise = true;
            return 1;
        }

        if (this->events->size() < 1000)
            return 1;

        ull itercount = 0;
        this->x_divider = this->y_divider = 1.0; 
        this->rot_divider = 10000;
        this->div_divider = 10000;

        clock_t begin = std::clock();
        
        std::string window_name = "Minimization output dynamic";
        if (this->manual_) {
            cv::namedWindow(window_name, cv::WINDOW_NORMAL);
        }


        this->iteration_step ();
        itercount ++;

        while (this->x_divider < 32 * 10 || 
               this->y_divider < 32 * 10 || 
               this->rot_divider < 32 * 1000 || 
               this->div_divider < 32 * 1000) {

        if (fabs(this->model.dx / this->x_divider)    < 1e-5 && 
            fabs(this->model.dy / this->y_divider)    < 1e-5 && 
            fabs(this->model.rot / this->rot_divider) < 1e-4 && 
            fabs(this->model.div / this->div_divider) < 1e-1) break;

            float old_dx  = this->model.dx;
            float old_dy  = this->model.dy;
            float old_rot = this->model.rot;
            float old_div = this->model.div;

            this->iteration_step ();
            itercount ++;

            if (max_itercount > 0 && int(itercount) > max_itercount) {
                break;
            }

            if (this->model.dx * old_dx < 0)   this->x_divider *= 2;
            if (this->model.dy * old_dy < 0)   this->y_divider *= 2;
            if (this->model.rot * old_rot < 0) this->rot_divider *= 2;
            if (this->model.div * old_div < 0) this->div_divider *= 2;

            
            if (this->manual_) {
                std::cout << this->model << std::endl;
                accel.writeout_events(this->events);
                while (cv::waitKey(33) != 32) { // 'space'
                    cv::imshow(window_name, EventFile::color_time_img(this->events, false));
                }
            }
        }

        accel.writeout_events(this->events);
        clock_t end = std::clock();

        if (VERBOSE)
            std::cout << "\tMinimization elapsed: " << double(end - begin) / CLOCKS_PER_SEC << " sec.\n"
                      << "\t\tIterations: " << itercount << " time per iteration: "
                      << (double(end - begin) / CLOCKS_PER_SEC) / itercount << "\n";

        for (auto &e : *this)
            e.assume_score(0); 

        return 0;
    }


    int manual () {
        this->manual_ = true;
        if ((this->scale_img_x < this->scale * RES_X / 15) && (this->scale_img_y < this->scale * RES_Y / 15)) {
                if (VERBOSE) std::cout << "Window size is too small; (" << this->metric_wsizex
                                       << ", " << this->metric_wsizey << "). Skipping...\n";
            return 1;
        }

        int value_x = 127, value_y = 127, value_rot = 127, value_div = 127, fine = 500;

        std::string window_name = "Minimization output";
        std::string window_name_color = "Minimization output color";
        std::string window_name_additional = "Minimization - additional";
        cv::namedWindow(window_name, cv::WINDOW_NORMAL);
        cv::namedWindow(window_name_color, cv::WINDOW_NORMAL);
        cv::namedWindow(window_name_additional, cv::WINDOW_NORMAL);

        cv::createTrackbar("x tilt", window_name, &value_x, 255, NULL);
        cv::createTrackbar("y tilt", window_name, &value_y, 255, NULL);
        cv::createTrackbar("rot", window_name, &value_rot, 255, NULL);
        cv::createTrackbar("div", window_name, &value_div, 255, NULL);
        cv::createTrackbar("fine/coarse", window_name, &fine, 1000, NULL);



        sll utmin = 9999999, utmax = 0;
        for (auto &e : *this) {
            if (e.t > utmax) utmax = e.t;
            if (e.t < utmin) utmin = e.t;
        }


        int code = 0;
        while (code != 27) { // 'esc'
            code = cv::waitKey(33);
            if (code == 99) { // 'c'
                this->run();

                cv::setTrackbarPos("x tilt", window_name, 127);
                cv::setTrackbarPos("y tilt", window_name, 127);
                cv::setTrackbarPos("rot", window_name, 127);
                cv::setTrackbarPos("div", window_name, 127);
            }


            if (code == 115) { // 's'
                cv::Mat time_img_s = accel.get_time_img(this->events, this->metric_wsizex, 
                                              this->metric_wsizey, this->scale,
                                              this->x_shift, this->y_shift);

                cv::normalize(time_img_s, time_img_s, 0, 255, cv::NORM_MINMAX, CV_8UC1); 
                cv::imwrite("./out/time_" + std::to_string(this->current_time) + ".jpg", time_img_s);
            }


            double nx = double(value_x - 127) / double(fine + 1);
            double ny = double(value_y - 127) / double(fine + 1);
            double rot = double(value_rot - 127) / double(fine + 1);
            double div = double(value_div - 127) / double(fine + 1);

            double cx = (model.cx - this->x_shift) / this->scale;
            double cy = (model.cy - this->y_shift) / this->scale;


            model.dx = nx;
            model.dy = ny;
            model.rot = rot;
            model.div = div;

            this->model.update_accumulators(10000, 10000, 1000, 1000);
            accel.project_4param_reinit<T>(this->events,
                             -model.total_dx, 
                             -model.total_dy, cx, cy, 
                             model.total_div,
                             -model.total_rot);

            accel.writeout_events(this->events);

            cv::Mat time_img = accel.get_time_img(this->events, this->metric_wsizex, 
                                                  this->metric_wsizey, this->scale,
                                                  this->x_shift, this->y_shift);
            accel.fast_model(this->model, time_img);

            cv::normalize(time_img, time_img, 0, 1, cv::NORM_MINMAX, CV_32FC1);
            //if (VERBOSE) std::cout << this->model << std::endl;

            //std::cout << "Times: " << current_time << "\t" << utmin << "\t" << utmax << "\n";


            cv::imshow(window_name, time_img);
            cv::imshow(window_name, this->get_gradient_img_color());
            //cv::imshow(window_name, EventFile::projection_img(this->events, this->scale, false));


            //cv::imshow(window_name, get_misalignment_img_color());

            cv::imshow(window_name_color, EventFile::color_time_img(this->events, this->scale, false));
            cv::imshow(window_name_additional, this->get_LR_gradient_img_color());
            //cv::imshow(window_name_additional, EventFile::projection_img(this->events, this->scale, false));
        }

        for (auto &e : *this)
            e.assume_score(0);

        return 0;
    }


    void set_maxiter(int val) {
        this->max_itercount = val;
    }

    
    inline void set_time (ull t_) {
        this->current_time = t_;
        for (auto &e : *this)
            e.set_local_time(this->current_time);
    }

 
    void set_cloud (T *events_, int sc_) {
        this->events = events_;
        this->scale = sc_;

        this->x_min = RES_X; this->y_min = RES_Y;
        this->x_max = 0; this->y_max = 0;

        for (auto &e : *this) {
            if ((int)e.fr_x > this->x_max) this->x_max = e.fr_x;
            if ((int)e.fr_y > this->y_max) this->y_max = e.fr_y;
            if ((int)e.fr_x < this->x_min) this->x_min = e.fr_x;
            if ((int)e.fr_y < this->y_min) this->y_min = e.fr_y;
            e.reset();
        }
     
        this->metric_wsizex = sc_ * (this->x_max - this->x_min);
        this->metric_wsizey = sc_ * (this->y_max - this->y_min);
        this->set_scale(this->scale);

        int w = this->metric_wsizex + this->scale;
        int h = this->metric_wsizey + this->scale;
        accel.init_gpu(this->events, w, h);
    }

    void set_scale (int sc_) {
        this->scale = sc_;
        assert(this->scale % 2 != 0);

        this->scale_img_x = this->metric_wsizex + this->scale;
        this->scale_img_y = this->metric_wsizey + this->scale;

        this->x_shift = - double((this->x_max - this->x_min) / 2 + this->x_min) * double(this->scale) 
                        + double(this->metric_wsizex) / 2.0 + this->scale / 2;
        this->y_shift = - double((this->y_max - this->y_min) / 2 + this->y_min) * double(this->scale) 
                        + double(this->metric_wsizey) / 2.0 + this->scale / 2;
    }

    ObjectModel get_model () {
        return this->model;
    }

    void set_model(ObjectModel m) {
        this->model = m;
        //double cx = (model.cx - this->x_shift) / this->scale;
        //double cy = (model.cy - this->y_shift) / this->scale;

        accel.project_4param_reinit<T>(this->events,
                             -model.total_dx, 
                             -model.total_dy, model.cx, model.cy, 
                             model.total_div,
                             -model.total_rot);
    }

    cv::Mat get_time_img() {return this->time_img; }
    cv::Mat get_sharpness_img() {return this->sharpness_img; }

private:
    void iteration_step () {
        cv::Mat time_img;
        
        accel.writeout_events(this->events);

        /*
        if (!accel.gpu_enabled) {
            time_img = accel.get_time_img(this->events, this->metric_wsizex, 
                                          this->metric_wsizey, this->scale,
                                          this->x_shift, this->y_shift);
        } else {
            time_img = accel.get_time_img_cpu(this->metric_wsizex, 
                                              this->metric_wsizey, this->scale,
                                              this->x_shift, this->y_shift);
        }
        */

        time_img = accel.get_time_img(this->events, this->metric_wsizex, 
                                      this->metric_wsizey, this->scale,
                                      this->x_shift, this->y_shift);


        accel.fast_model(this->model, time_img);
        this->model.update_accumulators(this->rot_divider, this->div_divider, this->x_divider, this->y_divider);

        double cx = (model.cx - this->x_shift) / this->scale;
        double cy = (model.cy - this->y_shift) / this->scale;

        /*
        accel.project_4param<T>(this->events,
                             -model.dx / this->x_divider, 
                             -model.dy / this->y_divider, cx, cy, 
                             model.div / this->div_divider,
                             -model.rot / this->rot_divider);         
        */
        accel.project_4param_reinit<T>(this->events,
                             -model.total_dx, 
                             -model.total_dy, cx, cy, 
                             model.total_div,
                             -model.total_rot);
        model.cx = cx;
        model.cy = cy;
    }


public:
    cv::Mat get_gradient_img () {
        cv::Mat time_img = accel.get_time_img_cpu(this->events, this->metric_wsizex, 
                                              this->metric_wsizey, this->scale,
                                              this->x_shift, this->y_shift); 
        cv::Mat grad_x, grad_y;
        cv::Mat abs_grad_x, abs_grad_y;

        cv::Mat pr_img = EventFile::projection_img(this->events, this->scale, false);
        
        //accel.LR_Sobel(time_img, 50, grad_x, grad_y);
        accel.LR_Sobel_fuse(time_img, pr_img, 50, grad_x, grad_y);

        EventFile::gradient_img_fuse(pr_img, grad_x, grad_y);

        cv::convertScaleAbs( grad_x, abs_grad_x );
        cv::convertScaleAbs( grad_y, abs_grad_y );

        /// Total Gradient (approximate)
        cv::Mat grad;
        cv::addWeighted(cv::abs(grad_x), 0.5, cv::abs(grad_y), 0.5, 0, grad );

        return grad;
    }

    cv::Mat get_gradient_img_color () {
        cv::Mat time_img = accel.get_time_img_cpu(this->events, this->metric_wsizex, 
                                                  this->metric_wsizey, this->scale,
                                                  this->x_shift, this->y_shift); 
        
        cv::Mat pr_img = EventFile::projection_img(this->events, this->scale, false);

        cv::Mat grad_x, grad_y;
        accel.Sobel(time_img, grad_x, grad_y);
        //EventFile::gradient_img_fuse(pr_img, grad_x, grad_y);

        return EventFile::color_gradient_img(grad_x, grad_y);
    }

    cv::Mat get_LR_gradient_img_color () {
        cv::Mat time_img = accel.get_time_img_cpu(this->events, this->metric_wsizex, 
                                                  this->metric_wsizey, this->scale,
                                                  this->x_shift, this->y_shift); 
        
        cv::Mat pr_img = EventFile::projection_img(this->events, this->scale, false);
        
        cv::Mat grad_x, grad_y;
        accel.LR_Sobel(time_img, 9, grad_x, grad_y);
        //accel.LR_Sobel_fuse(time_img, pr_img, 50, grad_x, grad_y);

        //EventFile::gradient_img_fuse(pr_img, grad_x, grad_y);
        return EventFile::color_gradient_img(grad_x, grad_y);
    }


    cv::Mat get_misalignment_img_color () {
        cv::Mat img = accel.get_time_img_cpu(this->events, this->metric_wsizex, 
                                             this->metric_wsizey, this->scale,
                                             this->x_shift, this->y_shift); 
         
        cv::Mat grad_x = cv::Mat::zeros(img.size(), CV_32FC1);
        //cv::Mat grad_y = cv::Mat::zeros(img.size(), CV_32FC1);

        int nRows = img.rows;
        int nCols = img.cols;

        tbb::parallel_for(tbb::blocked_range<int>(1, nRows - 1), [&](const tbb::blocked_range<int>& r) {
        for (int i = r.begin(); i < r.end(); ++i) { 
            float *px = grad_x.ptr<float>(i);
            //float *py = grad_y.ptr<float>(i);
            float *p = img.ptr<float>(i);
            for (int j = 1; j < nCols - 1; ++j) {
                if (p[j] <= 0.000001) continue;
                 
                px[j] = goto_min(img, j, i) + goto_max(img, j, i);
            }
        }
        });
      
        
        cv::normalize(grad_x, grad_x, 0, 255, cv::NORM_MINMAX, CV_8UC1); 
         
        return grad_x;
        //return EventFile::color_gradient_img(grad_x, grad_y);
    }


    static int goto_min(cv::Mat &img, int i_, int j_) {
        int nRows = img.rows;
        int nCols = img.cols;

        int ret = 1;
        int i = i_, j = j_;
        float val = img.at<float>(j, i);

        while (true) {
        int val_ref = val;
        int i_ref = i, j_ref = j;
        for (int k = 0; k < 3; ++k) {
            for (int l = 0; l < 3; ++l) {
                if (k == 1 && l == 1) continue;
                float val_ = img.at<float>(l + j - 1, k + i - 1);
                if (val_ <= 0.000001) continue;
                
                if (val_ref > val_) {
                    val_ref = val_; 
                    i_ref = k + i - 1;
                    j_ref = l + j - 1;
                }
            }
        }
        if (val_ref == val) {
            return ret;
        }

        i = i_ref;
        j = j_ref;
        val = val_ref;
        
        if ((i <= 0) || (j <= 0) || (i >= nRows - 1) || (j >= nCols - 1))
            return ret;
        
        ++ ret;
        }

    }

    static int goto_max(cv::Mat &img, int i_, int j_) {
        int nRows = img.rows;
        int nCols = img.cols;

        int ret = 1;
        int i = i_, j = j_;
        float val = img.at<float>(j, i);

        while (true) {
        int val_ref = val;
        int i_ref = i, j_ref = j;
        for (int k = 0; k < 3; ++k) {
            for (int l = 0; l < 3; ++l) {
                if (k == 1 && l == 1) continue;
                float val_ = img.at<float>(l + j - 1, k + i - 1);
                if (val_ <= 0.000001) continue;
                
                if (val_ref < val_) {
                    val_ref = val_; 
                    i_ref = k + i - 1;
                    j_ref = l + j - 1;
                }
            }
        }
        if (val_ref == val) {
            return ret;
        }

        i = i_ref;
        j = j_ref;
        val = val_ref;
        
        if ((i <= 0) || (j <= 0) || (i >= nRows - 1) || (j >= nCols - 1))
            return ret;
        
        ++ ret;
        }

    }

      
    static bool adj_minmax(cv::Mat &img, float ref, int i, int j, 
                                         int i_min, int j_min,
                                         int i_max, int j_max) {        
        float min = ref, max = ref;
        for (int k = 0; k < 3; ++k) {
            for (int l = 0; l < 3; ++l) {
                if (k == 1 && l == 1) continue;
                float val = img.at<float>(l + j - 1, k + i - 1);
                if (val <= 0.000001) continue;
                if (val < min) {
                    i_min = k + i - 1;
                    j_min = l + j - 1;
                    min = val;
                }

                if (val > max) {
                    i_max = k + i - 1;
                    j_max = l + j - 1;
                    max = val;
                }
            }
        }

        if (min == ref) {
            i_min = - 1;
            j_min = - 1;
        }

        if (min == ref) {
            i_min = - 1;
            j_min = - 1;
        }

        return true;
    }


protected:
    auto begin() {return this->events->begin(); }
    auto end() {return this->events->end(); }
};


#endif // OPTIMIZER_ROLLING_H
