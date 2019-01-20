#ifndef EVENT_H
#define EVENT_H

#include <better_flow/common.h>

class Cluster;
class Event {
public:
    uint fr_x, fr_y;
    sll t;
    ull timestamp;
    bool noise;
    bool valid;

    double pr_x, pr_y;
    double nx, ny, nz;
    double u, v;

    double best_u, best_v, max_score;
    double best_pr_x, best_pr_y;

    bool visited;
    Cluster *cl;
    int cl_id;

    Event () : fr_x(UINT_MAX), fr_y(UINT_MAX), t(ULLONG_MAX), timestamp(LLONG_MAX),
        noise(true), pr_x(NAN), pr_y(NAN), nx(0), ny(0), nz(NZ), u(0), v(0),
        best_u(0), best_v(0), max_score(0), best_pr_x(NAN), best_pr_y(NAN),
        visited(false), cl(NULL), cl_id(-1) {}

    Event (uint x_, uint y_, ull t_) : fr_x(x_), fr_y(y_), t(t_), timestamp(t_),
        noise(false), valid(false), pr_x(x_), pr_y(y_), nx(0), ny(0), nz(NZ), u(0), v(0),
        best_u(0), best_v(0), max_score(0), best_pr_x(x_), best_pr_y(y_),
        visited(false), cl(NULL), cl_id(-1) {}

    inline sll operator- (const Event& rhs) {
        return sll(this->timestamp) - sll(rhs.timestamp);
    }

    inline bool operator==(const Event& rhs) {  
        bool coord_eq = (this->fr_x == rhs.fr_x) && (this->fr_y == rhs.fr_y);
        ull dt = (this->timestamp >= rhs.timestamp) ? this->timestamp - rhs.timestamp : rhs.timestamp - this->t;
        bool time_eq = dt < 100000; // dt < 0.1 ms
        return coord_eq && time_eq;
    }

    inline bool operator!=(const Event& rhs) {
        return !(*this == rhs);
    }

    inline uint get_x () const {return this->fr_x; }
    inline uint get_y () const {return this->fr_y; }

    inline void reset() {
        this->pr_x = this->fr_x;
        this->pr_y = this->fr_y;
        this->nx = this->ny = 0;
        this->u = this->v = 0;
    }

    inline void set_local_time(ull t_) {
        this->t = (this->timestamp > t_) ? this->timestamp - t_ : -sll(t_ - this->timestamp);
    }

    inline void project (double nx_, double ny_, double nz_ = NZ) {
        this->nx = nx_;
        this->ny = ny_;
        this->nz = nz_;
        this->apply_project();
    }

    inline void project_dn (double dnx_, double dny_) {
        this->nx += dnx_;
        this->ny += dny_;
        this->apply_project();
    }

    inline void project_divcrl (double cx, double cy, double div, double crl) {
        cv::Point2d r(this->pr_x - cx, this->pr_y - cy);
       
        cv::Point2d r_(std::cos(crl) * r.x - std::sin(crl) * r.y,
                       std::sin(crl) * r.x + std::cos(crl) * r.y); 

        cv::Point2d dn = - r_ * div + (r_ - r);
        this->project_dn(dn.x, dn.y);
    }

    inline void project_4param (double dnx_, double dny_, double cx, double cy, double div, double crl) {
        cv::Point2d r(this->pr_x - cx, this->pr_y - cy);
       
        cv::Point2d r_(std::cos(crl) * r.x - std::sin(crl) * r.y,
                       std::sin(crl) * r.x + std::cos(crl) * r.y); 

        cv::Point2d dn = - r_ * div + (r_ - r);
        this->project_dn(dn.x + dnx_, dn.y + dny_);
    }


    inline void project_4param_reinit (double dnx_, double dny_, double cx, double cy, double div, double crl) {
        cv::Point2d r(this->pr_x - cx, this->pr_y - cy);
       
        cv::Point2d r_(std::cos(crl) * r.x - std::sin(crl) * r.y,
                       std::sin(crl) * r.x + std::cos(crl) * r.y); 

        cv::Point2d dn = - r_ * div + (r_ - r);

        this->nx = dn.x + dnx_;
        this->ny = dn.y + dny_;
        this->apply_project();
    }


    inline void apply_score(double score) {
        if (score > this->max_score) {
            this->max_score = score;
            this->best_u = this->u;
            this->best_v = this->v;
            this->best_pr_x = this->pr_x;
            this->best_pr_y = this->pr_y;
        }
    }

    inline void assume_score(double score) {
        this->max_score = score;
        this->best_u = this->u;
        this->best_v = this->v;
        this->best_pr_x = this->pr_x;
        this->best_pr_y = this->pr_y;
    }

    inline double n_from_u (double vel) {
        return vel * (this->nz / (1000000000/(T_DIVIDER * 10000)));
    }

    inline void compute_uv () {
        double xy_len = hypot(this->nx, this->ny);
        double speed = xy_len / (this->nz / (1000000000/(T_DIVIDER * 10000)));
        this->u = (xy_len == 0) ? 0 : speed * this->nx / xy_len;
        this->v = (xy_len == 0) ? 0 : speed * this->ny / xy_len;
        this->best_u = this->u;
        this->best_v = this->v;
    }

private:
    inline void apply_project () {
        
        /*
        double xy_len = hypot(this->nx, this->ny);
        double speed = xy_len / (this->nz / (1000000000/(T_DIVIDER * 10000)));
        this->u = (xy_len == 0) ? 0 : speed * this->nx / xy_len;
        this->v = (xy_len == 0) ? 0 : speed * this->ny / xy_len;

        //this->u = this->nx / (1000000000/(T_DIVIDER * 10000));
        //this->v = this->ny / (1000000000/(T_DIVIDER * 10000)); 

        double kx = this->nx / this->nz;
        double ky = this->ny / this->nz;
        
        this->pr_x = double(this->fr_x) - double(this->t / T_DIVIDER) / 10000 * kx;
        this->pr_y = double(this->fr_y) - double(this->t / T_DIVIDER) / 10000 * ky;
        */

        
        float kx = float(this->nx) / this->nz;
        float ky = float(this->ny) / this->nz;
        
        this->pr_x = float(this->fr_x) - kx * float(this->t) / 10000.0;
        this->pr_y = float(this->fr_y) - ky * float(this->t) / 10000.0;
    }
};


#endif // EVENT_H
