#ifndef ACCEL_LIB_H
#define ACCEL_LIB_H

#include <better_flow/event.h>
#include <better_flow/opencl_driver.h>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#if OPENCL_ENABLED
#include <CL/cl.hpp>
#endif

class AccelLib {
private:
#if OPENCL_ENABLED
    cl::Context *context;
    cl::Program *program;
    cl::CommandQueue *queue;

    cl::Kernel pr_4param, model_helper_ker;

    // OpenCL buffers
    cl::Buffer buffer_fr_x, buffer_fr_y, buffer_nx, buffer_ny, buffer_pr_x, buffer_pr_y, buffer_t;
#endif

    // Corresponding C buffers
    int *fr_x_c, *fr_y_c, *t_c;
    float *nx_c, *ny_c, *pr_x_c, *pr_y_c;
    size_t size; // Number of events

#if OPENCL_ENABLED
    // Improc OpenCL buffers
    cl::Buffer img_buf, grad_x_buf, grad_y_buf, rot_buf, div_buf;
#endif

    // Corresponding Mats
    cv::Mat grad_x, grad_y, rots, divs;

public:
    bool gpu_enabled;

public: 
    AccelLib () : fr_x_c(NULL), fr_y_c(NULL), t_c(NULL),
        nx_c(NULL), ny_c(NULL), pr_x_c(NULL), pr_y_c(NULL) {
        this->gpu_enabled = OpenCLDriver::enabled;

#if OPENCL_ENABLED
        this->context = OpenCLDriver::context;
        this->program = OpenCLDriver::program;        
        this->queue = OpenCLDriver::queue;
        this->pr_4param = OpenCLDriver::pr_4param_ker;
        this->model_helper_ker = OpenCLDriver::model_helper_ker;
#endif
    }

    ~AccelLib () { 
        this->clear_buffers();
    }

    inline void clear_buffers () {
        if (this->fr_x_c != NULL) delete this->fr_x_c;
        if (this->fr_y_c != NULL) delete this->fr_y_c;
        if (this->nx_c != NULL) delete this->nx_c;
        if (this->ny_c != NULL) delete this->ny_c;
        if (this->pr_x_c != NULL) delete this->pr_x_c;
        if (this->pr_y_c != NULL) delete this->pr_y_c;
        if (this->t_c != NULL) delete this->t_c;
    }

    template<class T> inline void init_gpu (T *events, int nRows, int nCols) {
        if (!this->gpu_enabled) return;

#if OPENCL_ENABLED
        // Input: fr_x, fr_y, nx, ny, t
        // Output: pr_x, pr_y
        // Params: dnx_, dny_, cx, cy, div, crl
        this->size = events->size();
        std::cout << "Initializing GPU bufers for " << this->size << " events " 
                  << "and (" << nRows << " x " << nCols << ") images" << std::endl;

        this->clear_buffers ();
        this->fr_x_c = new int[size];
        this->fr_y_c = new int[size];
        this->t_c    = new int[size];
        this->nx_c = new float[size];
        this->ny_c = new float[size];
        this->pr_x_c = new float[size];
        this->pr_y_c = new float[size];

        for (ull i = 0; i < size; ++i) {
            this->fr_x_c[i] = (*events)[i].fr_x;
            this->fr_y_c[i] = (*events)[i].fr_y;
            this->nx_c[i]   = (*events)[i].nx;
            this->ny_c[i]   = (*events)[i].ny;
            this->pr_x_c[i] = (*events)[i].pr_x;
            this->pr_y_c[i] = (*events)[i].pr_y;
            this->t_c[i]    = (*events)[i].t;
        }

        this->buffer_fr_x = cl::Buffer(*this->context, CL_MEM_WRITE_ONLY, sizeof(int) * size);
        this->buffer_fr_y = cl::Buffer(*this->context, CL_MEM_WRITE_ONLY, sizeof(int) * size);
        this->buffer_t    = cl::Buffer(*this->context, CL_MEM_WRITE_ONLY, sizeof(int) * size);
        this->buffer_nx   = cl::Buffer(*this->context, CL_MEM_READ_WRITE, sizeof(float) * size);
        this->buffer_ny   = cl::Buffer(*this->context, CL_MEM_READ_WRITE, sizeof(float) * size);
        this->buffer_pr_x = cl::Buffer(*this->context, CL_MEM_READ_WRITE, sizeof(float) * size);
        this->buffer_pr_y = cl::Buffer(*this->context, CL_MEM_READ_WRITE, sizeof(float) * size);

        this->queue->enqueueWriteBuffer(this->buffer_fr_x, CL_TRUE, 0, sizeof(int) * size,   this->fr_x_c);
        this->queue->enqueueWriteBuffer(this->buffer_fr_y, CL_TRUE, 0, sizeof(int) * size,   this->fr_y_c);
        this->queue->enqueueWriteBuffer(this->buffer_t,    CL_TRUE, 0, sizeof(int) * size,    this->t_c);
        this->queue->enqueueWriteBuffer(this->buffer_nx,   CL_TRUE, 0, sizeof(float) * size,   this->nx_c);
        this->queue->enqueueWriteBuffer(this->buffer_ny,   CL_TRUE, 0, sizeof(float) * size,   this->ny_c);
        this->queue->enqueueWriteBuffer(this->buffer_pr_x, CL_TRUE, 0, sizeof(float) * size, this->pr_x_c);
        this->queue->enqueueWriteBuffer(this->buffer_pr_y, CL_TRUE, 0, sizeof(float) * size, this->pr_y_c);

        cl_int err;
        err = pr_4param.setArg(0, this->buffer_fr_x);
        err = pr_4param.setArg(1, this->buffer_fr_y);
        err = pr_4param.setArg(2, this->buffer_nx);
        err = pr_4param.setArg(3, this->buffer_ny);
        err = pr_4param.setArg(4, this->buffer_pr_x);
        err = pr_4param.setArg(5, this->buffer_pr_y);
        err = pr_4param.setArg(6, this->buffer_t);
        if (err != 0) std::cout << "OpenCL Error: " << err << std::endl;

        this->img_buf = cl::Buffer(*this->context, CL_MEM_WRITE_ONLY, sizeof(float) * nRows * nCols);
        this->grad_x_buf = cl::Buffer(*this->context, CL_MEM_READ_ONLY, sizeof(float) * nRows * nCols);
        this->grad_y_buf = cl::Buffer(*this->context, CL_MEM_READ_ONLY, sizeof(float) * nRows * nCols);
        this->rot_buf = cl::Buffer(*this->context, CL_MEM_READ_ONLY, sizeof(float) * nRows * nCols);
        this->div_buf = cl::Buffer(*this->context, CL_MEM_READ_ONLY, sizeof(float) * nRows * nCols);

        err = model_helper_ker.setArg(0, this->img_buf);
        err = model_helper_ker.setArg(1, this->grad_x_buf);
        err = model_helper_ker.setArg(2, this->grad_y_buf);
        err = model_helper_ker.setArg(3, this->rot_buf);
        err = model_helper_ker.setArg(4, this->div_buf);
        if (err != 0) std::cout << "OpenCL Error: " << err << std::endl;

        this->grad_x = cv::Mat::zeros(nRows, nCols, CV_32FC1);
        this->grad_y = cv::Mat::zeros(nRows, nCols, CV_32FC1);
        this->rots = cv::Mat::zeros(nRows, nCols, CV_32FC1);
        this->divs = cv::Mat::zeros(nRows, nCols, CV_32FC1);
#endif
    }

    template<class T> static inline cv::Mat get_time_img_cpu (T *events, int w, int h, int scale, int x_sh, int y_sh) {
        cv::Mat project_img_avg = cv::Mat::zeros(w + scale, h + scale, CV_32FC1);
        cv::Mat project_img_cnt = cv::Mat::zeros(w + scale, h + scale, CV_32FC1);

        for (auto &e : *events) {
            if (e.noise) continue;

            int x = e.pr_x * scale + x_sh;
            int y = e.pr_y * scale + y_sh;

            if ((x >= w + scale / 2) || (x < scale / 2) || (y >= h + scale / 2) || (y < scale / 2))
                continue;
         
            for (int jx = x - scale / 2; jx <= x + scale / 2; ++jx) {
                for (int jy = y - scale / 2; jy <= y + scale / 2; ++jy) {
                    project_img_avg.at<float>(jx, jy) += double(e.t) / 1000000000.0;
                    project_img_cnt.at<float>(jx, jy) += 1;
                }
            }
        }
        
        tbb::parallel_for(tbb::blocked_range<int>(0, w + scale), [&](const tbb::blocked_range<int>& r) {
        for (int jx = r.begin(); jx < r.end(); ++jx) {
            for (int jy = 0; jy < h + scale; ++jy) {
                if (project_img_cnt.at<float>(jx, jy) < 1) continue;
                project_img_avg.at<float>(jx, jy) /= project_img_cnt.at<float>(jx, jy);
            }
        }
        });

        return project_img_avg;
    }


    inline cv::Mat get_time_img_cpu (int w, int h, int scale, int x_sh, int y_sh) {
        cv::Mat project_img_avg = cv::Mat::zeros(w + scale, h + scale, CV_32FC1);
        cv::Mat project_img_cnt = cv::Mat::zeros(w + scale, h + scale, CV_32FC1);

        for (ull i = 0; i < this->size; ++i) {
            int x = this->pr_x_c[i] * scale + x_sh;
            int y = this->pr_y_c[i] * scale + y_sh;

            if ((x >= w + scale / 2) || (x < scale / 2) || (y >= h + scale / 2) || (y < scale / 2))
                continue;
         
            for (int jx = x - scale / 2; jx <= x + scale / 2; ++jx) {
                for (int jy = y - scale / 2; jy <= y + scale / 2; ++jy) {
                    project_img_avg.at<float>(jx, jy) += double(this->t_c[i]) / 1000000000.0;
                    project_img_cnt.at<float>(jx, jy) += 1;
                }
            }
        }
            
        for (int jx = 0; jx < w + scale; ++jx) {
            for (int jy = 0; jy < h + scale; ++jy) {
                if (project_img_cnt.at<float>(jx, jy) < 1) continue;
                project_img_avg.at<float>(jx, jy) /= project_img_cnt.at<float>(jx, jy);
            }
        }

        return project_img_avg;
    }


    template<class T> inline cv::Mat get_time_img (T *events, int w, int h, int scale, int x_sh, int y_sh) {
        if (!this->gpu_enabled) {
            return this->get_time_img_cpu<T>(events, w, h, scale, x_sh, y_sh);
        }
        
        // GPU version is unstable
        return this->get_time_img_cpu<T>(events, w, h, scale, x_sh, y_sh);

#if OPENCL_ENABLED
        int w_true = w + scale;
        int h_true = h + scale;

        cv::Mat ret = cv::Mat::zeros(w_true, h_true, CV_32FC1);

        size_t size = events->size();
        for (ull i = 0; i < size; ++i) {
            this->pr_x_c[i] = (*events)[i].pr_x;
            this->pr_y_c[i] = (*events)[i].pr_y;
            this->t_c[i]    = float((*events)[i].t) / 1000000000.0;
        }

        this->queue->enqueueWriteBuffer(this->buffer_pr_x, CL_TRUE, 0, sizeof(float) * size, this->pr_x_c);
        this->queue->enqueueWriteBuffer(this->buffer_pr_y, CL_TRUE, 0, sizeof(float) * size, this->pr_y_c);
        this->queue->enqueueWriteBuffer(this->buffer_t,    CL_TRUE, 0, sizeof(float) * size,    this->t_c);
        
        cl::Buffer timg_buf(*this->context, CL_MEM_READ_WRITE, sizeof(float) * w_true * h_true);
        cl::Buffer cimg_buf(*this->context, CL_MEM_READ_WRITE, sizeof(float) * w_true * h_true);
        cl::Buffer img_add_buf(*this->context, CL_MEM_READ_WRITE, sizeof(float) * w_true * h_true);

        cl_int err;
        cl::Kernel time_img_ker = cl::Kernel(*this->program, "time_img");
        err = time_img_ker.setArg(0, this->buffer_pr_x);
        err = time_img_ker.setArg(1, this->buffer_pr_y);
        err = time_img_ker.setArg(2, this->buffer_t);
        err = time_img_ker.setArg(3, timg_buf); 
        err = time_img_ker.setArg(4, cimg_buf); 
        err = time_img_ker.setArg(5, img_add_buf); 
        err = time_img_ker.setArg(6, w);
        err = time_img_ker.setArg(7, h);
        err = time_img_ker.setArg(8, scale);
        err = time_img_ker.setArg(9, x_sh);
        err = time_img_ker.setArg(10, y_sh);
        if (err != 0) std::cout << "OpenCL Error: " << err << std::endl;

        this->queue->enqueueNDRangeKernel(time_img_ker, cl::NullRange, cl::NDRange(size), cl::NullRange);
        this->queue->finish();

        this->queue->enqueueReadBuffer(timg_buf, CL_TRUE, 0, sizeof(float) * w_true * h_true, ret.data);
        return ret;
#endif
    }

    template<class T> inline void project_4param_reinit (T *events, double dnx_, double dny_, double cx, double cy, double div, double crl) {
        for (auto &e : *events) {
            e.project_4param_reinit(dnx_, dny_, cx, cy, div, crl);
        }
    }

    template<class T> inline void project_4param_cpu (T *events, double dnx_, double dny_, double cx, double cy, double div, double crl) {
        for (auto &e : *events) {
            e.project_4param(dnx_, dny_, cx, cy, div, crl);
        }
    }

    template<class T> inline void project_4param (T *events, double dnx_, double dny_, double cx, double cy, double div, double crl) {
        if (!this->gpu_enabled) {
            this->project_4param_cpu<T>(events, dnx_, dny_, cx, cy, div, crl);
            return;
        }

#if OPENCL_ENABLED
        size_t size = events->size();

        for (ull i = 0; i < size; ++i) {;
            this->t_c[i]    = (*events)[i].t;
        }

        this->queue->enqueueWriteBuffer(this->buffer_t,    CL_TRUE, 0, sizeof(int) * size,    this->t_c);
        this->queue->finish();
        
        
        cl_int err;
        err = pr_4param.setArg(7, (float)dnx_);
        err = pr_4param.setArg(8, (float)dny_);
        err = pr_4param.setArg(9, (float)cx);
        err = pr_4param.setArg(10, (float)cy);
        err = pr_4param.setArg(11, (float)div);
        err = pr_4param.setArg(12, (float)crl);
        if (err != 0) std::cout << "OpenCL Error: " << err << std::endl;

        this->queue->enqueueNDRangeKernel(pr_4param, cl::NullRange, cl::NDRange(size), cl::NullRange);
        this->queue->finish();

        this->queue->enqueueReadBuffer(this->buffer_pr_x, CL_TRUE, 0, sizeof(float) * size, this->pr_x_c);     
        this->queue->enqueueReadBuffer(this->buffer_pr_y, CL_TRUE, 0, sizeof(float) * size, this->pr_y_c);   
        this->queue->finish();
#endif
    }

    template<class T> inline void writeout_events(T *events) {
        if (!this->gpu_enabled) {
            return;
        }

#if OPENCL_ENABLED
        size_t size = events->size();
        this->queue->enqueueReadBuffer(this->buffer_pr_x, CL_TRUE, 0, sizeof(float) * size, this->pr_x_c);     
        this->queue->enqueueReadBuffer(this->buffer_pr_y, CL_TRUE, 0, sizeof(float) * size, this->pr_y_c);      
        this->queue->enqueueReadBuffer(this->buffer_nx,   CL_TRUE, 0, sizeof(float) * size,   this->nx_c);     
        this->queue->enqueueReadBuffer(this->buffer_ny,   CL_TRUE, 0, sizeof(float) * size,   this->ny_c);     
        this->queue->finish();
        for (ull i = 0; i < size; ++i) {
            (*events)[i].pr_x = this->pr_x_c[i];
            (*events)[i].pr_y = this->pr_y_c[i];
            (*events)[i].nx = this->nx_c[i];
            (*events)[i].ny = this->ny_c[i];
        }
#endif
    }

    ObjectModel fast_model(cv::Mat &time_img) {
        ObjectModel model;
        this->fast_model(model, time_img);
        return model;
    }

    void fast_model(ObjectModel &model, cv::Mat &time_img) {
        if (!this->gpu_enabled) {
            model.update(time_img);
            return;
        }

#if OPENCL_ENABLED
        model.center_of_mass(time_img);
        model.dx = 0;
        model.dy = 0;
        model.rot = 0;
        model.div = 0;
        model.cnt = 0;

        int nRows = time_img.rows;
        int nCols = time_img.cols;

        this->queue->enqueueWriteBuffer(this->img_buf, CL_TRUE, 0, sizeof(float) * nRows * nCols, time_img.data);

        cl_int err;
        err = model_helper_ker.setArg(5, (float)model.cx);
        err = model_helper_ker.setArg(6, (float)model.cy);
        err = model_helper_ker.setArg(7, nCols);
        err = model_helper_ker.setArg(8, nRows);
        if (err != 0) std::cout << "OpenCL Error: " << err << std::endl;

        this->queue->enqueueNDRangeKernel(model_helper_ker, cl::NullRange, cl::NDRange(nRows * nCols), cl::NullRange);
        this->queue->finish();

        this->queue->enqueueReadBuffer(this->grad_x_buf, CL_TRUE, 0, sizeof(float) * nRows * nCols, this->grad_x.data);
        this->queue->enqueueReadBuffer(this->grad_y_buf, CL_TRUE, 0, sizeof(float) * nRows * nCols, this->grad_y.data);
        this->queue->enqueueReadBuffer(this->rot_buf, CL_TRUE, 0, sizeof(float) * nRows * nCols, this->rots.data);
        this->queue->enqueueReadBuffer(this->div_buf, CL_TRUE, 0, sizeof(float) * nRows * nCols, this->divs.data);
        this->queue->finish();

        float* p  = (float *)time_img.data;
        float* gx = (float *)(this->grad_x.data);
        float* gy = (float *)(this->grad_y.data);
        float* rt = (float *)(this->rots.data);
        float* dv = (float *)(this->divs.data);
    
        for(long int i = 0; i < nRows * nCols; ++i) {
            if (*p > 0.000001) {
                model.rot += *rt;
                model.div += *dv;
                model.dx += *gx;
                model.dy += *gy;
                model.cnt ++;
            }
            p ++;
            gx ++;
            gy ++;
            rt ++;
            dv ++;
        }

        model.rot /= (double)model.cnt;
        model.div /= (double)model.cnt;       
        model.dx /= (double)model.cnt;
        model.dy /= (double)model.cnt;
#endif
    }

    void Sobel(cv::Mat &img, cv::Mat &grad_x, cv::Mat &grad_y) {
        if (!this->gpu_enabled) {
            this->Sobel_cpu(img, grad_x, grad_y);
            return;
        }

#if OPENCL_ENABLED
        grad_x = cv::Mat::zeros(img.size(), CV_32FC1);
        grad_y = cv::Mat::zeros(img.size(), CV_32FC1);
        
  	    int nRows = img.rows;
        int nCols = img.cols;

        cl::Buffer img_buf(*this->context, CL_MEM_READ_WRITE, sizeof(float) * nRows * nCols);
        cl::Buffer grad_x_buf(*this->context, CL_MEM_READ_WRITE, sizeof(float) * nRows * nCols);
        cl::Buffer grad_y_buf(*this->context, CL_MEM_READ_WRITE, sizeof(float) * nRows * nCols);

        this->queue->enqueueWriteBuffer(img_buf, CL_TRUE, 0, sizeof(float) * nRows * nCols, img.data);

        cl_int err;
        cl::Kernel sharr_ker = cl::Kernel(*this->program, "sharr");
        err = sharr_ker.setArg(0, img_buf);
        err = sharr_ker.setArg(1, grad_x_buf);
        err = sharr_ker.setArg(2, grad_y_buf);
        err = sharr_ker.setArg(3, nCols);
        err = sharr_ker.setArg(4, nRows);
        if (err != 0) std::cout << "OpenCL Error: " << err << std::endl;

        this->queue->enqueueNDRangeKernel(sharr_ker, cl::NullRange, cl::NDRange(nRows * nCols), cl::NullRange);
        this->queue->finish();

        this->queue->enqueueReadBuffer(grad_x_buf, CL_TRUE, 0, sizeof(float) * nRows * nCols, grad_x.data);
        this->queue->enqueueReadBuffer(grad_y_buf, CL_TRUE, 0, sizeof(float) * nRows * nCols, grad_y.data);
#endif
    }

    static void LR_Sobel_fuse(cv::Mat &img, cv::Mat &pr_img, int wsize, cv::Mat &grad_x, cv::Mat &grad_y) {
        grad_x = cv::Mat::zeros(img.size(), CV_32FC1);
        grad_y = cv::Mat::zeros(img.size(), CV_32FC1);

        cv::Mat grad_x_hres, grad_y_hres;
        AccelLib::Sobel_cpu(img, grad_x_hres, grad_y_hres);
        EventFile::gradient_img_fuse(pr_img, grad_x_hres, grad_y_hres);

        int nRows = img.rows;
        int nCols = img.cols;

        tbb::parallel_for(tbb::blocked_range<int>(wsize / 2, nRows - wsize / 2), [&](const tbb::blocked_range<int>& r) {
        for (int i = r.begin(); i < r.end(); ++i) { 
            float *px = grad_x.ptr<float>(i);
            float *py = grad_y.ptr<float>(i);
            //float *p = img.ptr<float>(i);
            for (int j = wsize / 2; j < nCols - wsize / 2; ++j) {
                //if (p[j] <= 0.000001) continue;
                float dx = 0, dy = 0;
                if (AccelLib::LR_sobel_point(j, i, wsize, grad_x_hres, dx)) {
                    px[j] = dx;
                }
                if (AccelLib::LR_sobel_point(j, i, wsize, grad_y_hres, dy)) {
                    py[j] = dy;
                } 
            }
        }
        });
    }

    static void LR_Sobel(cv::Mat &img, int wsize, cv::Mat &grad_x, cv::Mat &grad_y) {
        grad_x = cv::Mat::zeros(img.size(), CV_32FC1);
        grad_y = cv::Mat::zeros(img.size(), CV_32FC1);

        cv::Mat grad_x_hres, grad_y_hres;
        AccelLib::Sobel_cpu(img, grad_x_hres, grad_y_hres);

        int nRows = img.rows;
        int nCols = img.cols;

        tbb::parallel_for(tbb::blocked_range<int>(wsize / 2, nRows - wsize / 2), [&](const tbb::blocked_range<int>& r) {
        for (int i = r.begin(); i < r.end(); ++i) { 
            float *px = grad_x.ptr<float>(i);
            float *py = grad_y.ptr<float>(i);
            //float *p = img.ptr<float>(i);
            for (int j = wsize / 2; j < nCols - wsize / 2; ++j) {
                //if (p[j] <= 0.000001) continue;
                float dx = 0, dy = 0;
                if (AccelLib::LR_sobel_point(j, i, wsize, grad_x_hres, dx)) {
                    px[j] = dx;
                }
                if (AccelLib::LR_sobel_point(j, i, wsize, grad_y_hres, dy)) {
                    py[j] = dy;
                } 
            }
        }
        });
    }

    static bool LR_sobel_point(int i, int j, int wsize, cv::Mat &img, float &ret) {
        ret = 0;
        unsigned int cnt = 0;
        for (int k = 0; k < wsize; ++k) {
            for (int l = 0; l < wsize; ++l) {
                float val = img.at<float>(l + j - wsize/2, k + i - wsize/2);
                if (fabs(val) > 1e-8) {
                    cnt ++;
                    ret += val;
                }
            }
        }
        if (cnt < wsize * wsize / 4) return false;
        ret = (cnt == 0) ? 0 : ret / (float)cnt;
        return true;
    }


    static void Sobel_cpu(cv::Mat &img, cv::Mat &grad_x, cv::Mat &grad_y) {
        //cv::Scharr(img, grad_y, CV_32FC1, 1, 0, 1, 0, cv::BORDER_DEFAULT);
        //cv::Scharr(img, grad_x, CV_32FC1, 0, 1, 1, 0, cv::BORDER_DEFAULT);
        //cv::Sobel(img, grad_y, CV_32FC1, 1, 0, 9, 1, 0, cv::BORDER_DEFAULT);
        //cv::Sobel(img, grad_x, CV_32FC1, 0, 1, 9, 1, 0, cv::BORDER_DEFAULT);
        //grad_x = -grad_x;
        //grad_y = -grad_y;
        //return;

        grad_x = cv::Mat::zeros(img.size(), CV_32FC1);
        grad_y = cv::Mat::zeros(img.size(), CV_32FC1);

        int nRows = img.rows;
        int nCols = img.cols;

        tbb::parallel_for(tbb::blocked_range<int>(1, nRows - 1), [&](const tbb::blocked_range<int>& r) {
        for (int i = r.begin(); i < r.end(); ++i) { 
            float *px = grad_x.ptr<float>(i);
            float *py = grad_y.ptr<float>(i);
            float *p = img.ptr<float>(i);
            for (int j = 1; j < nCols - 1; ++j) {
                if (p[j] <= 0.000001) continue;
                float dx = 0, dy = 0;
                if (AccelLib::sobel_point(img, j, i, dx, dy)) {
                    px[j] = dx;
                    py[j] = dy;
                }
            }
        }
        });
    }

    static bool sobel_point(cv::Mat &img, int i, int j, float &dx, float &dy) {
        int sharr_x [] = {3, 0, -3, 10, 0, -10, 3, 0, -3};
        int sharr_y [] = {3, 10, 3, 0, 0, 0, -3, -10, -3};
        int mask_x [] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
        int mask_y [] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    
        if ((img.at<float>(j - 1, i - 1) <= 0.000001) || 
            (img.at<float>(j + 1, i - 1) <= 0.000001)) {
            mask_x[0] = 0;
            mask_x[2] = 0;
        }

        if ((img.at<float>(j - 1, i) <= 0.000001) || 
            (img.at<float>(j + 1, i) <= 0.000001)) {
            mask_x[3] = 0;
            mask_x[5] = 0;
        }

        if ((img.at<float>(j - 1, i + 1) <= 0.000001) || 
            (img.at<float>(j + 1, i + 1) <= 0.000001)) {
            mask_x[6] = 0;
            mask_x[8] = 0;
        }

        if ((img.at<float>(j - 1, i - 1) <= 0.000001) || 
            (img.at<float>(j - 1, i + 1) <= 0.000001)) {
            mask_y[0] = 0;
            mask_y[6] = 0;
        }

        if ((img.at<float>(j, i - 1) <= 0.000001) || 
            (img.at<float>(j, i + 1) <= 0.000001)) {
            mask_y[1] = 0;
            mask_y[7] = 0;
        }

        if ((img.at<float>(j + 1, i - 1) <= 0.000001) || 
            (img.at<float>(j + 1, i + 1) <= 0.000001)) {
            mask_y[2] = 0;
            mask_y[8] = 0;
        }

        float x_norm = 0, y_norm = 0;
        for (int k = 0; k < 9; ++k) {
            x_norm += abs(sharr_x[k]) * mask_x[k];
            y_norm += abs(sharr_y[k]) * mask_y[k];
        }


        int idx = 0;
        dx = dy = 0;
        for (int k = 0; k < 3; ++k) {
            for (int l = 0; l < 3; ++l) {
                float val = img.at<float>(l + j - 1, k + i - 1);
                if (val <= 0.000001) return false;
                
                dx += val * sharr_x[idx];
                dy += val * sharr_y[idx];
                idx++;
            }
        }

        /*
        x_norm *= 32;
        y_norm *= 32;
        if (x_norm > 0) dx /= x_norm;
        if (y_norm > 0) dy /= y_norm;
        */

        return true;
    }
};


#endif // ACCEL_LIB_H
