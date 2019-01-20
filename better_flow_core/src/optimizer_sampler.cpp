#include <better_flow/optimizer_sampler.h>


int OptimizerLocal::run () {
    this->nx = 0, this->ny = 0;
    this->last_score = 0, this->dscore = 0;
    this->dnx = 0.01, this->dny = 0.01, this->dn_th = (NZ * T_DIVIDER * 1000.0) / (10 * this->scale * FROM_MS(MAX_TIME_MS));

    if ((this->scale_img_x < this->scale * RES_X / 15) && (this->scale_img_y < this->scale * RES_Y / 15)) {
        if (VERBOSE) std::cout << "Window size is too small; (" << this->metric_wsizex
                               << ", " << this->metric_wsizey << "). Skipping...\n";
        return 1;
    }

    clock_t begin = std::clock();
    this->last_score = this->iteration_step(this->nx, this->ny);
     
    // std::string window_name = "Minimization output dynamic";
    // cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    while (hypot(dnx, dny) > dn_th) {
        nx = this->compute_new_nx(nx);
        ny = this->compute_new_ny(ny);

        /*    
        cv::normalize(project_img, project_img, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::imshow(window_name, project_img);
        cv::displayStatusBar(window_name, "primary avg: " + std::to_string(last_score));
        std::cout << nx << " " << ny << " " << dnx << " " << dny << "\n";
        while (cv::waitKey(33) != 32) {} // 'space'
        */
    }

    clock_t end = std::clock();

    if (VERBOSE)
        std::cout << "\tMinimization elapsed: " << double(end - begin) / CLOCKS_PER_SEC << " sec.\n";
    return 0;
}


int OptimizerLocal::manual () {
    this->nx = 0, this->ny = 0;
    this->last_score = 0, this->dscore = 0;
    this->dnx = 0.01, this->dny = 0.01, this->dn_th = (NZ * T_DIVIDER * 1000.0) / (10 * this->scale * FROM_MS(MAX_TIME_MS));

    if ((this->scale_img_x < this->scale * RES_X / 15) && (this->scale_img_y < this->scale * RES_Y / 15)) {
        if (VERBOSE) std::cout << "Window size is too small; (" << this->metric_wsizex
                               << ", " << this->metric_wsizey << "). Skipping...\n";
        return 1;
    }

    this->last_score = this->iteration_step(this->nx, this->ny);
    int value_x = 127, value_y = 127, fine = 500;

    std::string window_name = "Minimization output";
    std::string window_name_color = "Minimization output color";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::namedWindow(window_name_color, cv::WINDOW_NORMAL);

    cv::createTrackbar("x tilt", window_name, &value_x, 255, NULL);
    cv::createTrackbar("y tilt", window_name, &value_y, 255, NULL);
    cv::createTrackbar("fine/coarse", window_name, &fine, 1000, NULL);

    int code = 0;
    while (code != 27) { // 'esc'
        code = cv::waitKey(33);
        if (code == 99) { // 'c'
            this->run();

            cv::setTrackbarPos("x tilt", window_name, this->nx * double(fine + 1) + 127);
            cv::setTrackbarPos("y tilt", window_name, this->ny * double(fine + 1) + 127);
        }

        this->nx = double(value_x - 127) / double(fine + 1);
        this->ny = double(value_y - 127) / double(fine + 1);

        this->last_score = this->iteration_step(this->nx, this->ny);

        cv::normalize(project_img, project_img, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::displayStatusBar(window_name, "primary avg: " + std::to_string(last_score));
        cv::imshow(window_name, project_img);
        cv::imshow(window_name_color, EventFile::color_time_img(this->events));
        std::cout << nx << " " << ny << " " << dnx << " " << dny << "\n";
    }

    return 0;
}


double OptimizerLocal::compute_new_nx(double nx_) {
    double nx_new = nx_ + this->dnx;
    double new_score = this->iteration_step(nx_new, this->ny);

    this->dscore = new_score - this->last_score;
    this->last_score = new_score;
    
    if (this->dscore <= 0) {    
        this->dnx = - this->dnx / 2.0;
    }

    return nx_new;
}


double OptimizerLocal::compute_new_ny(double ny_) {
    double ny_new = ny_ + this->dny;
    double new_score = this->iteration_step(this->nx, ny_new);

    this->dscore = new_score - this->last_score;
    this->last_score = new_score;
    
    if (this->dscore <= 0) {    
        this->dny = - this->dny / 2.0;
    }

    return ny_new;
}


inline double OptimizerLocal::iteration_step (double nx_, double ny_) {
    for (auto &e : *this) e.project(nx_, ny_);
    this->event_c.project(nx_, ny_);

    this->project_img = cv::Mat::zeros(this->scale_img_x, this->scale_img_y, CV_8UC1);

    double x_shift = - this->event_c.pr_x * this->scale + double(this->metric_wsizex) / 2.0;
    double y_shift = - this->event_c.pr_y * this->scale + double(this->metric_wsizey) / 2.0;

    for (auto &e : *this) {
        int x = e.pr_x * this->scale + x_shift;
        int y = e.pr_y * this->scale + y_shift;

        if ((x >= this->metric_wsizex) || (x < 0) || (y >= this->metric_wsizey) || (y < 0))
            continue;

        x += this->scale / 2;
        y += this->scale / 2;
        
        for (int jx = x - this->scale / 2; jx <= x + this->scale / 2; ++jx) {
            for (int jy = y - this->scale / 2; jy <= y + this->scale / 2; ++jy) {
                if (this->project_img.at<uchar>(jx, jy) < 255) {
                    this->project_img.at<uchar>(jx, jy) ++;
                }
            }
        }
    }

    if (this->scale > 1) {
        cv::GaussianBlur(this->project_img, this->project_img, cv::Size(this->scale, this->scale), 0, 0);
    }

    return this->get_event_score();
}


inline double OptimizerLocal::iteration_step_debug (double nx_, double ny_) {
    for (auto &e : *this) e.project(nx_, ny_);
    this->event_c.project(nx_, ny_);

    this->project_img = cv::Mat::zeros(this->scale_img_x, this->scale_img_y, CV_8UC1);

    double x_shift = - this->event_c.pr_x * this->scale + double(this->metric_wsizex) / 2.0;
    double y_shift = - this->event_c.pr_y * this->scale + double(this->metric_wsizey) / 2.0;

    double nz_avg = 0;
    long int nz_avg_cnt = 0;
    for (auto &e : *this) {
        int x = e.pr_x * this->scale + x_shift;
        int y = e.pr_y * this->scale + y_shift;

        if ((x >= this->metric_wsizex) || (x < 0) || (y >= this->metric_wsizey) || (y < 0))
            continue;

        x += this->scale / 2;
        y += this->scale / 2;
        
        for (int jx = x - this->scale / 2; jx <= x + this->scale / 2; ++jx) {
            for (int jy = y - this->scale / 2; jy <= y + this->scale / 2; ++jy) {
                nz_avg += 1;
                if (this->project_img.at<uchar>(jx, jy) == 0) {
                    nz_avg_cnt++;
                    this->project_img.at<uchar>(jx, jy) = 1;
                }
            }
        }
    }

    return (nz_avg_cnt == 0) ? 0 : nz_avg / double(nz_avg_cnt);
}


inline double OptimizerLocal::get_event_score () {
    // Average of nonzero
    double nz_avg = 0;
    long int nz_avg_cnt = 0;
    uchar* p = this->project_img.data;
    for(int i = 0; i < this->project_img.rows * this->project_img.cols; ++i, p++) {
        if (*p == 0) continue;
        nz_avg_cnt ++;
        nz_avg += *p;
    }
    nz_avg = (nz_avg_cnt == 0) ? 0 : nz_avg / double(nz_avg_cnt);
    return nz_avg;
}


void OptimizerLocal::update_fields () {
    assert(this->scale % 2 != 0);

    this->scale_img_x = this->metric_wsizex + this->scale;
    this->scale_img_y = this->metric_wsizey + this->scale;

    this->project_img = cv::Mat::zeros(this->scale_img_x, this->scale_img_y, CV_8UC1);
}
