#include <better_flow/optimizer_global.h>


void OptimizerGlobal::project_all (double nx_, double ny_, double nz_) {

    // Project events
    clock_t begin_pr = std::clock();
    for (auto &e : *this) e.project(nx_, ny_, nz_);
    clock_t end_pr = std::clock();

    // Generate scaled projection image
    clock_t begin_img = std::clock();
    this->project_img = cv::Mat::zeros(scale_bordered_img_x, scale_bordered_img_y, CV_8UC1);
    for (auto &e : *this) {
        int x = e.pr_x * this->scale - this->events->x_min * this->scale;
        int y = e.pr_y * this->scale - this->events->y_min * this->scale;

        if ((x >= this->scale_img_x - this->scale) || (x < 0) || (y >= this->scale_img_y - this->scale) || (y < 0))
            continue;

        x += this->scale / 2 + this->metric_wsize / 2;
        y += this->scale / 2 + this->metric_wsize / 2;
        
        for (int jx = x - this->scale / 2; jx <= x + this->scale / 2; ++jx) {
            for (int jy = y - this->scale / 2; jy <= y + this->scale / 2; ++jy) {
                if (this->project_img.at<uchar>(jx, jy) < 255)
                    this->project_img.at<uchar>(jx, jy) ++;
            }
        }
    }

    if (this->scale > 1) {
        cv::GaussianBlur(this->project_img, this->project_img, cv::Size(this->scale, this->scale), 0, 0);
    }
    clock_t end_img = std::clock();


    // Compute a per-projected-pixel sharpness score 
    clock_t begin_score = std::clock();
    this->current_scores = cv::Mat::zeros(scale_img_x, scale_img_y, CV_32FC1);

    // - parallel version - 
    tbb::parallel_for(tbb::blocked_range<int>(0, this->events->size()), [&](const tbb::blocked_range<int>& r) {
        for (int i = r.begin(); i < r.end(); ++i) {
            auto &e = (*this->events)[i];

            int x = e.pr_x * this->scale - this->events->x_min * this->scale;
            int y = e.pr_y * this->scale - this->events->y_min * this->scale;

            if ((x >= this->scale_img_x - this->scale) || (x < 0) || (y >= this->scale_img_y - this->scale) || (y < 0))
                continue;

            if (this->current_scores.at<float>(x, y) != 0) continue;

            this->current_scores.at<float>(x, y) = 
                this->get_event_score(x + this->metric_wsize / 2 + this->scale / 2, y + this->metric_wsize / 2 + this->scale / 2);
        }
    });
    clock_t end_score = std::clock();

    // Apply the new score
    clock_t begin_apply = std::clock();


    for (auto &e : *this) {
        int x = e.pr_x * this->scale - this->events->x_min * this->scale;
        int y = e.pr_y * this->scale - this->events->y_min * this->scale;

        if ((x >= this->scale_img_x - this->scale) || (x < 0) || (y >= this->scale_img_y - this->scale) || (y < 0))
            continue;
                    
        e.apply_score(this->current_scores.at<float>(x, y));
    }

    clock_t end_apply = std::clock();
    
    if (VERBOSE)
        std::cout << "\t Elapsed: " << double(end_apply - begin_pr) / CLOCKS_PER_SEC << " sec. ("
                  << "Projection: " << double(end_pr - begin_pr) / CLOCKS_PER_SEC << " sec. "
                  << "Pr image: " << double(end_img - begin_img) / CLOCKS_PER_SEC << " sec. "
                  << "Scoring: " << double(end_score - begin_score) / CLOCKS_PER_SEC << " sec. "
                  << "Applying best score: " << double(end_apply - begin_apply) / CLOCKS_PER_SEC << " sec.)\n";
} 


inline double OptimizerGlobal::get_event_score (int x, int y) {
    // Average of nonzero
    double nz_avg = 0;
    long int nz_avg_cnt = 0;

    for (int j = x - this->metric_wsize / 2; j <= x + this->metric_wsize / 2; ++j) {
        for (int i = y - this->metric_wsize / 2; i <= y + this->metric_wsize / 2; ++i) {
            if (this->project_img.at<uchar>(j, i) <= 0) continue;
            nz_avg_cnt ++;
            nz_avg += this->project_img.at<uchar>(j, i);
        }
    }

    nz_avg = (nz_avg_cnt == 0) ? 0 : nz_avg / double(nz_avg_cnt);
    return nz_avg;
}


void OptimizerGlobal::compute_flow_bruteforce () {
    // Unused; just for experiments
    double x_low = -0.09, x_hi = 0.09;
    double y_low = -0.04, y_hi = 0.04;
    double x_step = 0.001, y_step = 0.001;

    assert (x_low < x_hi && y_low < y_hi && x_step > 0 && y_step > 0);

    ull x_iterations = fabs(x_hi - x_low) / x_step + 1;
    ull y_iterations = fabs(y_hi - y_low) / y_step + 1;
    ull iterations = x_iterations * y_iterations;
    ull iteration_id = 0;

    double u_step = x_step / NZ * (1000000000 / (T_DIVIDER * 10000));
    double u_low = x_low / NZ * (1000000000 / (T_DIVIDER * 10000));
    double u_hi = x_hi / NZ * (1000000000 / (T_DIVIDER * 10000));
    double v_step = y_step / NZ * (1000000000 / (T_DIVIDER * 10000));
    double v_low = y_low / NZ * (1000000000 / (T_DIVIDER * 10000));
    double v_hi = y_hi / NZ * (1000000000 / (T_DIVIDER * 10000));
    
    std::cout << "u = [" << u_low << " ... " << u_hi << "] px/sec\t"
              << "(x = [" << x_low << " ... " << x_hi << "])" << std::endl;
    std::cout << "v = [" << v_low << " ... " << v_hi << "] px/sec\t"
              << "(y = [" << y_low << " ... " << y_hi << "])" << std::endl;
    std::cout << "u step = " << u_step << " px/sec\t(x step = " << x_step << ")" << std::endl;
    std::cout << "v step = " << v_step << " px/sec\t(y step = " << y_step << ")" << std::endl;
    std::cout << "iterations: x = " << x_iterations << "\ty = " << y_iterations
              << "\ttotal = " << iterations << std::endl;
    std::cout << std::endl;

    int precision = 3; // percent precicion; max is 5
    for (double nx = x_low; nx < x_hi; nx += x_step) {
        for (double ny = y_low; ny < y_hi; ny += y_step) {
            double percent = double(std::round(double(iteration_id * 100000) / double(iterations))) / 1000;
            if (int(percent * 100000) % ((int)std::round(std::pow(10, 5 - precision))) == 0)
                std::cout << "\rProcessing: " << std::fixed << std::setprecision(precision) << percent << "%" << std::flush;

            this->project_all (nx, ny);
            iteration_id ++;
        }
    }

    std::cout << "\rProcessing: " << std::fixed << std::setprecision(precision) << 100.0 << "%" << std::flush;
    std::cout << std::endl;
}


int OptimizerGlobal::manual (double nx_, double ny_) {
    if ((this->scale_img_x < this->scale * RES_X / 15) && (this->scale_img_y < this->scale * RES_Y / 15)) {
        if (VERBOSE) std::cout << "Window size is too small; (" << this->scale_img_x
                               << ", " << this->scale_img_y << "). Skipping...\n";
        return 1;
    }

    int value_x = 127, value_y = 127, fine = 999;

    std::string window_name = "Minimization output";
    std::string window_name_color = "Minimization output color";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);

    cv::createTrackbar("x tilt", window_name, &value_x, 255, NULL);
    cv::createTrackbar("y tilt", window_name, &value_y, 255, NULL);
    cv::createTrackbar("fine/coarse", window_name, &fine, 1000, NULL);

    double nx = nx_, ny = ny_;
    cv::setTrackbarPos("x tilt", window_name, nx * double(fine + 1) + 127);
    cv::setTrackbarPos("y tilt", window_name, ny * double(fine + 1) + 127);

    int code = 0;
    while (code != 27) { // 'esc'
        code = cv::waitKey(33);
  
        nx = double(value_x - 127) / double(fine + 1);
        ny = double(value_y - 127) / double(fine + 1);

        this->project_all (nx, ny);

        cv::normalize(project_img, project_img, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::imshow(window_name, project_img);
        cv::imshow(window_name_color, EventFile::color_time_img(this->events, true));
        std::cout << nx << " " << ny << "\n";
    }

    return 0;
}


void OptimizerGlobal::update_fields () {
    assert(this->scale % 2 != 0);
    assert(this->metric_wsize % 2 != 0);

    this->scale_img_x = (this->events->x_max - this->events->x_min + 1) * this->scale;
    this->scale_img_y = (this->events->y_max - this->events->y_min + 1) * this->scale;
    this->scale_bordered_img_x = this->scale_img_x + this->metric_wsize;
    this->scale_bordered_img_y = this->scale_img_y + this->metric_wsize;

    this->project_img = cv::Mat::zeros(this->scale_bordered_img_x, 
                                       this->scale_bordered_img_y, CV_8UC1);

    this->current_scores = cv::Mat::zeros(this->scale_img_x, 
                                          this->scale_img_y, CV_32FC1);
}
