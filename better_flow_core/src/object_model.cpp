#include <better_flow/object_model.h>
#include <better_flow/accel_lib.h>

void ObjectModel::compute (cv::Mat &time_img) {
    cv::Mat grad_x, grad_y;
    AccelLib::Sobel_cpu(time_img, grad_x, grad_y);

    this->dx = 0;
    this->dy = 0;    
    this->rot = 0;
    this->div = 0;
    this->cnt = 0;

    int nRows = grad_x.rows;
    int nCols = grad_x.cols;

    for(int i = 0; i < nRows; ++i) {
        float *px = grad_x.ptr<float>(i);
        float *py = grad_y.ptr<float>(i);
        float *p = time_img.ptr<float>(i);
        for (int j = 0; j < nCols; ++j) {
            if (p[j] > 0.000001) {
                cv::Point2d r(double(i) - this->cx, double(j) - this->cy);
                cv::Point2d g(px[j], py[j]);

                this->dx += px[j];
                this->dy += py[j];
                this->rot += r.cross(g);
                this->div += r.ddot(g);
                this->cnt ++;
            }
        }
    }

    this->rot /= double(this->cnt);
    this->div /= double(this->cnt);    
    this->dx /= double(this->cnt);
    this->dy /= double(this->cnt);
}


void ObjectModel::compute (cv::Mat &time_img, LinearEventCloud *events,
                           float scale, float x_shift, float y_shift, float p) {
    std::default_random_engine generator;
    uint n_samples = float(events->size() - 1) * p;
    std::uniform_int_distribution<int> distribution(0, (events->size() - 1));
    auto dice = std::bind (distribution, generator);
    
    float *smp_x = new float[n_samples];
    float *smp_y = new float[n_samples];
    float *smp_u = new float[n_samples];
    float *smp_v = new float[n_samples];

    this->dx = 0;
    this->dy = 0;

    this->cnt = 0;
    while (this->cnt < n_samples) {
        float dx_ = 0, dy_ = 0;
        auto &e = (*events)[dice()];

        int x = e.pr_x * scale + x_shift;
        int y = e.pr_y * scale + y_shift;

        if (ObjectModel::sobel_point(time_img, y, x, dx_, dy_)) {
            smp_x[this->cnt] = x;
            smp_y[this->cnt] = y;
            smp_u[this->cnt] = dx_;
            smp_v[this->cnt] = dy_;
            this->dx += dx_;
            this->dy += dy_;
            this->cnt ++;
        }
    }

    this->dx /= this->cnt;
    this->dy /= this->cnt;

    this->rot = 0;
    this->div = 0;

    for (uint i = 0; i < this->cnt; ++i) {
        smp_u[i] -= this->dx;
        smp_v[i] -= this->dy;

        cv::Point2d r(smp_x[i] - this->cx, smp_y[i] - this->cy);
        cv::Point2d g(smp_u[i], smp_v[i]);
        this->rot += r.cross(g);
        this->div += r.ddot(g);
    }

    this->rot /= double(this->cnt);
    this->div /= double(this->cnt);

    delete [] smp_x;
    delete [] smp_y;
    delete [] smp_u;
    delete [] smp_v;
}



void ObjectModel::center_of_mass (cv::Mat &time_img) {
    this->cx = 0;
    this->cy = 0;
    this->cnt = 0;

    int nRows = time_img.rows;
    int nCols = time_img.cols;

    for(int i = 0; i < nRows; ++i) {
        float *p = time_img.ptr<float>(i);
        for (int j = 0; j < nCols; ++j) {
            if (p[j] > 0.000001) {
                this->cx += i;
                this->cy += j;
                this->cnt ++;
            }
        }
    }

    assert(this->cnt > 0);

    this->cx /= double(this->cnt);
    this->cy /= double(this->cnt);
}


bool ObjectModel::sobel_point(cv::Mat &img, int i, int j, float &dx, float &dy) {
    int sharr_x [] = {3, 0, -3, 10, 0, -10, 3, 0, -3};
    int sharr_y [] = {3, 10, 3, 0, 0, 0, -3, -10, -3};
   
    int idx = 0;
    dx = dy = 0;
    for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
            float val = img.at<float>(l + j - 1, k + i -1);
            if (val <= 0.000001) return false;
            dx += val * sharr_x[idx];
            dy += val * sharr_y[idx];
            idx++;
        }
    }

    return true;
}
