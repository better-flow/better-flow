#include <better_flow/event_file.h>


cv::Mat EventFile::color_gradient_img (cv::Mat gx, cv::Mat gy) {
    if (gx.size() != gy.size()) {
        std::cout << "gx and gy sizes do not match!\n";
        return gx;
    }

    cv::Mat flow_hsv(gx.size(), CV_8UC3, cv::Scalar(0, 0, 0)); 
 
    int nRows = flow_hsv.rows;
    int nCols = flow_hsv.cols;

    int cnt = 0;
    double avg_speed = 0;
    for(int i = 0; i < nRows; ++i) {
        for (int j = 0; j < nCols; ++j) {
            double u = gx.at<float>(i, j);
            double v = gy.at<float>(i, j);
            double speed = hypot(u, v);
            if (speed != 0) {
                avg_speed += speed;
                cnt ++;
            }
        }
    }
    avg_speed /= cnt;

    for(int i = 0; i < nRows; ++i) {
        for (int j = 0; j < nCols; ++j) {
            double u = gx.at<float>(i, j);
            double v = gy.at<float>(i, j);

            double speed = 127 * hypot(u, v) / avg_speed; // Normalize
            double angle = 0;
            if (speed != 0)
                angle = (atan2(v, u) + 3.1416) * 180 / 3.1416;
                // angle = (atan2(v, u)) * 180;

            flow_hsv.at<cv::Vec3b>(i, j)[0] = angle/2;
            flow_hsv.at<cv::Vec3b>(i, j)[1] = (speed == 0) ? 0 : 255;
            flow_hsv.at<cv::Vec3b>(i, j)[2] = (speed == 0) ? 0 : speed;
        }
    }

    cv::Mat flow_bgr;

    #if CV_MAJOR_VERSION == 2
        cv::cvtColor(flow_hsv, flow_bgr, CV_HSV2BGR);
    #elif CV_MAJOR_VERSION >= 3
        cv::cvtColor(flow_hsv, flow_bgr, cv::COLOR_HSV2BGR);
    #endif

    return flow_bgr;
}

void EventFile::gradient_img_fuse (cv::Mat pr_img, cv::Mat &gx, cv::Mat &gy) {
    if (gx.size() != gy.size() || gx.size() != pr_img.size()) {
        std::cout << "gx / gy / pr_img sizes do not match!\n";
        std::cout << "\t" << gx.size() << " " << gy.size() << " " << pr_img.size() << "\n";
    }

    int nRows = pr_img.rows;
    int nCols = pr_img.cols;

    for(int i = 0; i < nRows; ++i) {
        for (int j = 0; j < nCols; ++j) {
            double u = gx.at<float>(i, j);
            double v = gy.at<float>(i, j);
            double speed = hypot(u, v);

            u = (speed == 0) ? 0 : u / speed;
            v = (speed == 0) ? 0 : v / speed;

            if (speed != 0) {
                speed = 255 - pr_img.at<uchar>(i, j);
            }

            u *= speed;
            v *= speed;

            gx.at<float>(i, j) = u;
            gy.at<float>(i, j) = v;
        }
    }
}


cv::Mat EventFile::generate_color_circle (bool show) {
    cv::Mat flow_hsv(4000, 4000, CV_8UC3, cv::Scalar(0, 0, 255));
    for (double u = -200; u <= 200; u += 0.1) {
        for (double v = -200; v <= 200; v += 0.1) {
            double speed = hypot(u, v);
            double angle = 0;
            if (speed != 0)
                angle = (atan2(v, u) + 3.1416) * 180 / 3.1416;

            flow_hsv.at<cv::Vec3b>((u + 200) * 10, (v + 200) * 10)[0] = angle / 2;
            flow_hsv.at<cv::Vec3b>((u + 200) * 10, (v + 200) * 10)[1] = speed;
        }
    }

    cv::Mat flow_bgr;

    #if CV_MAJOR_VERSION == 2
        cv::cvtColor(flow_hsv, flow_bgr, CV_HSV2BGR);
    #elif CV_MAJOR_VERSION >= 3
        cv::cvtColor(flow_hsv, flow_bgr, cv::COLOR_HSV2BGR);
    #endif

    if (show) {
        std::string window_name = "Color Circle";
        cv::namedWindow(window_name, cv::WINDOW_NORMAL);
        cv::imshow(window_name, flow_bgr);
    }

    return flow_bgr;
}


/*
void EventFile::evaluate (std::string fname) {
    std::cout << "Reading ground truth from file... (" << fname << ")" << std::endl << std::flush;

    std::vector<std::vector<std::pair<double,double>>> flow_gt;
    for (uint i = 0; i < RES_X; ++i) {
        std::vector<std::pair<double,double>> col;
        for (uint j = 0; j < RES_Y; ++j) {
            col.push_back(std::make_pair(nan(""), nan("")));
        }
        flow_gt.push_back(col);
    }

    std::ifstream flow_file(fname, std::ifstream::in);

    int x = 0, y = 0;
    double fx = 0, fy = 0;
    while (flow_file >> y >> x >> fy >> fx) {
        flow_gt[RES_X - x][y - 1].first  = fy;
        flow_gt[RES_X - x][y - 1].second = -fx;
    }

    flow_file.close();
    std::cout << "Finished reading ground truth" << std::endl << std::flush;

    double spd_diff_slice = 0;
    double ang_diff_slice = 0;
    double vec_diff_slice = 0;
    double end_diff_slice = 0;

    ull flows_in_slice = 0;

    for (auto &e : *this) {
        if (e.noise) continue;

        int x = e.best_pr_x;
        int y = e.best_pr_y;

        if ((x >= RES_X) || (x < 0) || (y >= RES_Y) || (y < 0))
            continue;

        flows_in_slice ++;

        // Estimated flow (normal)
        double dx_est = e.best_u;
        double dy_est = e.best_v;
        double est_vel = hypot(dx_est, dy_est);

        // Ground truth
        double dx_gt_full = flow_gt[x][y].first;
        double dy_gt_full = flow_gt[x][y].second;
    
        // Project full flow vector on the estimated vector
        double dx_gt = dx_gt_full;
        double dy_gt = dy_gt_full;
        if (est_vel != 0) {
            double nx = dx_est / est_vel;
            double ny = dy_est / est_vel;
            double vel = nx * dx_gt_full + ny * dy_gt_full;
            dx_gt = nx * vel;
            dy_gt = ny * vel;
        }        
        double gt_vel = hypot(dx_gt,  dy_gt);

        // ------------------------------------------------
        // Error calculation:
        // ------------------------------------------------

        // Speed difference (for current slice)
        spd_diff_slice += fabs(gt_vel - est_vel);

        // Angular difference (for current slice)
        double ang_diff_cos = 0;
        if ((gt_vel >= 0.00001) && (est_vel >= 0.00001)) {
            ang_diff_cos = (dx_gt * dx_est + dy_gt * dy_est) / (gt_vel * est_vel);
            if (ang_diff_cos >  1.0) ang_diff_cos =  1.0;
            if (ang_diff_cos < -1.0) ang_diff_cos = -1.0;
        }
        ang_diff_slice += acos(ang_diff_cos);

        // Speed vector difference (for current slice)
        vec_diff_slice += hypot(dx_gt - dx_est, dy_gt - dy_est);

        // Endpoint error (for current slice)
        double end_diff_cos = (dx_gt * dx_est + dy_gt * dy_est + 1) / 
                               sqrt((dx_gt * dx_gt + dy_gt * dy_gt + 1) * 
                              (dx_est * dx_est + dy_est * dy_est + 1));
        if (end_diff_cos >  1.0) end_diff_cos =  1.0;
        if (end_diff_cos < -1.0) end_diff_cos = -1.0;
        end_diff_slice += acos(end_diff_cos);      
    }


    std::cout << "\tSpeed difference (for current slice):\t" << spd_diff_slice / flows_in_slice << std::endl << std::flush;
    std::cout << "\tAngular difference (for current slice):\t" << ang_diff_slice / flows_in_slice << std::endl << std::flush;
    std::cout << "\tVector difference (for current slice):\t" << vec_diff_slice / flows_in_slice << std::endl << std::flush;
    std::cout << "\tEndpoint error (for current slice):\t" << end_diff_slice / flows_in_slice << std::endl << std::flush;
    std::cout << "--------------------------------------------" << std::endl << std::endl << std::flush;


    int scale = 100;
    cv::Mat flow_img(RES_X * scale, RES_Y * scale, CV_8UC3, cv::Scalar(0, 0, 0));

    for (auto &e : *this) {
        if (e.noise) continue;

        int x = e.best_pr_x;
        int y = e.best_pr_y;

        if ((x >= RES_X) || (x < 0) || (y >= RES_Y) || (y < 0))
            continue;

        // Estimated flow (normal)
        double dx_est = e.best_u / 15;
        double dy_est = e.best_v / 15;
        double est_vel = hypot(dx_est, dy_est);

        // Ground truth
        double dx_gt_full = flow_gt[x][y].first;
        double dy_gt_full = flow_gt[x][y].second;
    
        // Project full flow vector on the estimated vector
        double dx_gt = dx_gt_full;
        double dy_gt = dy_gt_full;
        if (est_vel != 0) {
            double nx = dx_est / est_vel;
            double ny = dy_est / est_vel;
            double vel = nx * dx_gt_full + ny * dy_gt_full;
            dx_gt = nx * vel;
            dy_gt = ny * vel;
        }

        int x0 = x * scale;
        int y0 = y * scale;
        int x1 = x0 + dx_est * scale;
        int y1 = y0 + dy_est * scale;
        int x2 = x0 + dx_gt_full * scale;
        int y2 = y0 + dy_gt_full * scale;
        int x3 = x0 + dx_gt * scale;
        int y3 = y0 + dy_gt * scale;

      
    #if CV_MAJOR_VERSION == 2 // No arrowedLine in opencv 2.4
        cv::line(flow_img, cv::Point(x0, y0), cv::Point(x2, y2), CV_RGB(255,0,0));
        cv::line(flow_img, cv::Point(x0, y0), cv::Point(x3, y3), CV_RGB(0,0,255));
        cv::line(flow_img, cv::Point(x0, y0), cv::Point(x1, y1), CV_RGB(255,255,255));
    #elif CV_MAJOR_VERSION == 3
        cv::arrowedLine(flow_img, cv::Point(x0, y0), cv::Point(x2, y2), CV_RGB(255,0,0));
        cv::arrowedLine(flow_img, cv::Point(x0, y0), cv::Point(x3, y3), CV_RGB(0,0,255));
        cv::arrowedLine(flow_img, cv::Point(x0, y0), cv::Point(x1, y1), CV_RGB(255,255,255));
    #endif
    }

    std::string window_name = "Flow comparison";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::imshow(window_name, flow_img);
}
*/


double EventFile::nonzero_average (cv::Mat img) {
    // Average of nonzero
    double nz_avg = 0;
    long int nz_avg_cnt = 0;
    uchar* p = img.data;
    for(int i = 0; i < img.rows * img.cols; ++i, p++) {
        if (*p == 0) continue;
        nz_avg_cnt ++;
        nz_avg += *p;
    }
    nz_avg = (nz_avg_cnt == 0) ? 0 : nz_avg / double(nz_avg_cnt);
    return nz_avg;
}
