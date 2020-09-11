#ifndef EVENT_FILE_H
#define EVENT_FILE_H

#include <better_flow/common.h>
#include <better_flow/event.h>
#include <better_flow/clustering.h>


class EventFile {
public:
    template<class T> static void from_file(T *events, std::string fname, double maxt);
    template<class T> static void from_file_uv(T *events, std::string fname, double maxt);
    template<class T> static void to_file_uv(T *events, std::string fname, double maxt);
    template<class T> static void from_file(T *events, std::string fname);
    template<class T> static void from_file_uv(T *events, std::string fname);
    template<class T> static void to_file_uv(T *events, std::string fname);

    template<class T> static void display_uvscore       (T *events);
    template<class T> static cv::Mat color_clusters_img (T *events, bool show_final = false);
    template<class T> static cv::Mat color_time_img     (T *events, int scale = 0, bool show_final = false);
    template<class T> static cv::Mat arrow_flow_img     (T *events);
    template<class T> static cv::Mat color_flow_img     (T *events);
    template<class T> static cv::Mat projection_img     (T *events, int scale = 1, bool show_final = false, double min_t = 0, 
                                                         double max_t = 0);
    template<class T> static cv::Mat projection_img_unopt (T *events, int scale);
    static cv::Mat color_gradient_img     (cv::Mat gx, cv::Mat gy);
    static void gradient_img_fuse (cv::Mat pr_img, cv::Mat &gx, cv::Mat &gy);
    static cv::Mat generate_color_circle (bool show = false);

    static double nonzero_average (cv::Mat img);
};


template<class T> void EventFile::from_file(T *events, std::string fname, double maxt) {
    std::cout << "Reading from file... (" << fname << ")" << std::endl << std::flush;
    std::ifstream event_file(fname, std::ifstream::in);

    ull cnt = 0;
    double t = 0;
    uint x = 0, y = 0;
    bool p = false;

    double t_hi = maxt;
    double t_low = maxt - 0.1;

    double t_0 = 0; // the earliest timestamp in the file
    clock_t begin = std::clock();
    
    event_file >> t_0 >> x >> y >> p;
    while (event_file >> t >> x >> y >> p) {
        if (t - t_0 > t_low)
            break;
    }

    while (event_file >> t >> x >> y >> p) {
        t -= t_0;
        if (t > t_hi)
            break;
    
        events->push_back(Event(y, x, FROM_SEC(t)));
        cnt ++;
    }
    clock_t end = std::clock();

    event_file.close();

    if (cnt == 0) {
        std::cout << "Read " << cnt << " events, finished" << std::endl << std::endl << std::flush;
        return;
    }

    std::cout << "Read " << cnt << " events, finished" << std::endl << std::flush;
    std::cout << "Elapsed: " << double(end - begin) / CLOCKS_PER_SEC << " sec." << std::endl << std::flush;
}


template<class T> void EventFile::from_file_uv(T *events, std::string fname, double maxt) {
    std::cout << "Reading ground truth from file... (" << fname << ")" << std::endl << std::flush;
    std::ifstream event_file(fname, std::ifstream::in);

    ull cnt = 0;
    double t = 0, u = 0, v = 0;
    uint x = 0, y = 0;
    bool p = false;

    double t_hi = maxt;
    double t_low = maxt - 0.1;

    double t_0 = 0; // the earliest timestamp in the file
    clock_t begin = std::clock();
 
    event_file >> t_0 >> x >> y >> p >> u >> v;
    while (event_file >> t >> x >> y >> p >> u >> v) {
        if (t - t_0 > t_low)
            break;
    }

    while (!event_file.eof()) {

        if (!(event_file >> t >> x >> y >> p >> u >> v)) {
            // std::cout << "error reading" << u << v << std::endl;
            continue;
        }

        t -= t_0;
        if (t > t_hi)
            break;
        
        Event e(y, x, FROM_SEC(t));
        double nx = e.n_from_u(v);
        double ny = e.n_from_u(u);

        e.project(nx, ny);
        e.compute_uv();

        if ((std::fabs(e.u - v) > 0.00001) || (std::fabs(e.v - u) > 0.00001)) {
            std::cout << "U and V do not match! Skipping...\n";
            std::cout << "\t" << e.u << " " << e.v << " vs " << v << " " << u << "\n";
            continue;
        }
        
        e.assume_score(100);

        events->push_back(e);
        cnt ++;
    }
    clock_t end = std::clock();

    event_file.close();

    if (cnt == 0) {
        std::cout << "Read " << cnt << " events, finished" << std::endl << std::endl << std::flush;
        return;
    }

    std::cout << "Read " << cnt << " events, finished" << std::endl << std::flush;
    std::cout << "Elapsed: " << double(end - begin) / CLOCKS_PER_SEC << " sec." << std::endl << std::flush;
}


template<class T> void EventFile::from_file(T *events, std::string fname) {
    std::cout << "Reading from file... (" << fname << ")" << std::endl << std::flush;
    std::ifstream event_file(fname, std::ifstream::in);

    ull cnt = 0;
    double t = 0;
    uint x = 0, y = 0;
    bool p = false;


    double t_0 = 0; // the earliest timestamp in the file
    clock_t begin = std::clock();
    
    if (event_file >> t_0 >> x >> y >> p) {
        events->push_back(Event(y, x, FROM_SEC(0)));
        cnt ++;
    }

    while (event_file >> t >> x >> y >> p) {
        t -= t_0;

        events->push_back(Event(y, x, FROM_SEC(t)));
        cnt ++;
    }
    clock_t end = std::clock();

    event_file.close();

    if (cnt == 0) {
        std::cout << "Read " << cnt << " events, finished" << std::endl << std::endl << std::flush;
        return;
    }

    std::cout << "Read " << cnt << " events, finished" << std::endl << std::flush;
    std::cout << "Elapsed: " << double(end - begin) / CLOCKS_PER_SEC << " sec." << std::endl << std::flush;
}


template<class T> void EventFile::from_file_uv(T *events, std::string fname) {
    std::cout << "Reading ground truth from file... (" << fname << ")" << std::endl << std::flush;
    std::ifstream event_file(fname, std::ifstream::in);

    ull cnt = 0;
    double t = 0, u = 0, v = 0;
    uint x = 0, y = 0;
    bool p = false;

    double t_0 = 0; // the earliest timestamp in the file
    clock_t begin = std::clock();
 
    event_file >> t_0 >> x >> y >> p >> u >> v;
    while (!event_file.eof()) {

        if (!(event_file >> t >> x >> y >> p >> u >> v)) {
            event_file.clear(); // Reset error state
            event_file.ignore(3); // This assumes we only fail on NaN.

            // std::cout << "error reading" << u << v << std::endl;
            continue;
        }

        t -= t_0;

        Event e(y, x, FROM_SEC(t));

        double nx = e.n_from_u(v);
        double ny = e.n_from_u(u);

        e.project(nx, ny);
        e.compute_uv();

        if ((std::fabs(e.u - v) > 0.00001) || (std::fabs(e.v - u) > 0.00001)) {
            std::cout << "U and V do not match! Skipping...\n";
            std::cout << "\t" << e.u << " " << e.v << " vs " << v << " " << u << "\n";
            continue;
        }

        e.assume_score(100);

        events->push_back(e);
        cnt ++;
    }
    clock_t end = std::clock();

    event_file.close();

    if (cnt == 0) {
        std::cout << "Read " << cnt << " events, finished" << std::endl << std::endl << std::flush;
        return;
    }

    std::cout << "Read " << cnt << " events, finished" << std::endl << std::flush;
    std::cout << "Elapsed: " << double(end - begin) / CLOCKS_PER_SEC << " sec." << std::endl << std::flush;
}



template<class T> void EventFile::to_file_uv(T *events, std::string fname, double maxt) {
    std::cout << "Writing events and flow to file... (" << fname << ")" << std::endl << std::flush;
    std::ofstream event_file(fname, std::ofstream::out);

    ull cnt = 0;
    clock_t begin = std::clock();
  
    // Mind that x and y (as well as u and v) are swapped
    for (auto &e : *events) {
        event_file << std::fixed << std::setprecision(9) << double(e.timestamp) / 1000000000 + maxt << " " << e.fr_y << " " << e.fr_x 
                   << " " << 1 << " " << e.best_v << " " << e.best_u << std::endl;
        cnt ++;
    }
    clock_t end = std::clock();

    event_file.close();

    if (cnt == 0) {
        std::cout << "Written " << cnt << " events, finished" << std::endl << std::endl << std::flush;
        return;
    }

    std::cout << "Written " << cnt << " events, finished" << std::endl << std::flush;
    std::cout << "Elapsed: " << double(end - begin) / CLOCKS_PER_SEC << " sec." << std::endl << std::flush;
}


template<class T> void EventFile::to_file_uv(T *events, std::string fname) {
    std::cout << "Writing events and flow to file... (" << fname << ")" << std::endl << std::flush;
    std::ofstream event_file(fname, std::ofstream::out);

    ull cnt = 0;
    clock_t begin = std::clock();
  
    // Mind that x and y (as well as u and v) are swapped
    for (auto &e : *events) {
        event_file << std::fixed << std::setprecision(9) << double(e.timestamp) / 1000000000 << " " << e.fr_y << " " << e.fr_x 
                   << " " << 1 << " " << e.best_v << " " << e.best_u << std::endl;
        cnt ++;
    }
    clock_t end = std::clock();

    event_file.close();

    if (cnt == 0) {
        std::cout << "Written " << cnt << " events, finished" << std::endl << std::endl << std::flush;
        return;
    }

    std::cout << "Written " << cnt << " events, finished" << std::endl << std::flush;
    std::cout << "Elapsed: " << double(end - begin) / CLOCKS_PER_SEC << " sec." << std::endl << std::flush;
}


template<class T> cv::Mat EventFile::arrow_flow_img (T *events) {
    uint scale_arrow_flow = 10;
    cv::Mat flow_arrow(RES_X * scale_arrow_flow, RES_Y * scale_arrow_flow, CV_8UC3, cv::Scalar(255, 255, 255));

    for (auto &e : *events) {
        if (e.noise) continue;

        int x = e.best_pr_x;
        int y = e.best_pr_y;

        if ((x >= RES_X) || (x < 0) || (y >= RES_Y) || (y < 0))
            continue;
 
    #if CV_MAJOR_VERSION == 2 // No arrowedLine in opencv 2.4
        cv::line(flow_arrow, cv::Point(y * scale_arrow_flow, x * scale_arrow_flow), 
                        cv::Point((y + e.best_v / 20) * scale_arrow_flow, (x + e.best_u / 20) * scale_arrow_flow), CV_RGB(0,0,255));
    #elif CV_MAJOR_VERSION >= 3
        cv::arrowedLine(flow_arrow, cv::Point(y * scale_arrow_flow, x * scale_arrow_flow), 
                        cv::Point((y + e.best_v / 20) * scale_arrow_flow, (x + e.best_u / 20) * scale_arrow_flow), CV_RGB(0,0,255));
    #endif
    }

    return flow_arrow;
}


template<class T> cv::Mat EventFile::color_flow_img (T *events) {
    cv::Mat flow_hsv(RES_X, RES_Y, CV_8UC3, cv::Scalar(0, 0, 255));

    for (auto &e : *events) {
        if (e.noise) continue;

        int x = e.best_pr_x;
        int y = e.best_pr_y;

        if ((x >= RES_X) || (x < 0) || (y >= RES_Y) || (y < 0))
            continue;

        double speed = hypot(e.best_u, e.best_v);
        double angle = 0;
        if (speed != 0)
            angle = (atan2(e.best_v, e.best_u) + 3.1416) * 180 / 3.1416;

        double log_spd = std::min(255.0, std::log(speed) / std::log(1.025));

        flow_hsv.at<cv::Vec3b>(x, y)[0] = angle / 2;
        flow_hsv.at<cv::Vec3b>(x, y)[1] = log_spd;
    }

    cv::Mat flow_bgr;

    #if CV_MAJOR_VERSION == 2
        cv::cvtColor(flow_hsv, flow_bgr, CV_HSV2BGR);
    #elif CV_MAJOR_VERSION >= 3
        cv::cvtColor(flow_hsv, flow_bgr, cv::COLOR_HSV2BGR);
    #endif

    return flow_bgr;
}


template<class T> void EventFile::display_uvscore (T *events) {
    std::string best_pr_hires_window_name = "Best Projected Hi Res";
    cv::namedWindow(best_pr_hires_window_name, cv::WINDOW_NORMAL);
    std::string fl_window_name = "Flow";
    cv::namedWindow(fl_window_name, cv::WINDOW_NORMAL);
    std::string fl_arrow_window_name = "Flow Arrow";
    cv::namedWindow(fl_arrow_window_name, cv::WINDOW_NORMAL);

 
    double scale = 15;
    int scale_img_x = (RES_X + 1) * scale;
    int scale_img_y = (RES_Y + 1) * scale;

    cv::Mat best_project_img = cv::Mat::zeros(RES_X, RES_Y, CV_8UC1);   
    for (auto &e : *events) {
        int x = e.best_pr_x;
        int y = e.best_pr_y;

        if ((x >= RES_X) || (x < 0) || (y >= RES_Y) || (y < 0))
            continue;

        if (best_project_img.at<uchar>(x, y) < 255)
            best_project_img.at<uchar>(x, y) ++;
    }

    cv::Mat best_project_hires_img = cv::Mat::zeros(scale_img_x, scale_img_y, CV_8UC1);
    for (auto &e : *events) {
        int x = e.best_pr_x * scale;
        int y = e.best_pr_y * scale;

        if ((x >= scale * RES_X) || (x < 0) || (y >= scale * RES_Y) || (y < 0))
            continue;

        x += scale / 2;
        y += scale / 2;
        
        for (int jx = x - scale / 2; jx <= x + scale / 2; ++jx) {
            for (int jy = y - scale / 2; jy <= y + scale / 2; ++jy) {
                if (best_project_hires_img.at<uchar>(jx, jy) < 255)
                    best_project_hires_img.at<uchar>(jx, jy) ++;
            }
        }
    }

    if (scale > 1) {
        cv::GaussianBlur(best_project_hires_img, best_project_hires_img, cv::Size(scale, scale), 0, 0);
    }
    //cv::normalize(best_project_hires_img, best_project_hires_img, 0, 255, cv::NORM_MINMAX, CV_8UC1); 
    double img_scale = 127.0 / EventFile::nonzero_average(best_project_hires_img);
    cv::convertScaleAbs(best_project_hires_img, best_project_hires_img, img_scale, 0);
    
    cv::adaptiveThreshold(best_project_img, best_project_img, 255, 
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, scale, 0);

    cv::Mat flow_hsv(RES_X, RES_Y, CV_8UC3, cv::Scalar(0, 0, 255));

    uint scale_arrow_flow = 10;
    cv::Mat flow_arrow(flow_hsv.rows * scale_arrow_flow, flow_hsv.cols * scale_arrow_flow, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat scores = cv::Mat::zeros(RES_X, RES_Y, CV_32FC1);

    for (auto &e : *events) {
        if (e.noise) continue;

        int x = e.best_pr_x;
        int y = e.best_pr_y;

        if ((x >= RES_X) || (x < 0) || (y >= RES_Y) || (y < 0))
            continue;

        scores.at<float>(x, y) = e.max_score;

        double speed = hypot(e.best_u, e.best_v);
        double angle = 0;
        if (speed != 0)
            angle = (atan2(e.best_v, e.best_u) + 3.1416) * 180 / 3.1416;

        flow_hsv.at<cv::Vec3b>(x, y)[0] = angle / 2;
        flow_hsv.at<cv::Vec3b>(x, y)[1] = speed;
      
    #if CV_MAJOR_VERSION == 2 // No arrowedLine in opencv 2.4
        cv::line(flow_arrow, cv::Point(y * scale_arrow_flow, x * scale_arrow_flow), 
                        cv::Point((y + e.best_v / 20) * scale_arrow_flow, (x + e.best_u / 20) * scale_arrow_flow), CV_RGB(0,0,255));
    #elif CV_MAJOR_VERSION == 3
        cv::arrowedLine(flow_arrow, cv::Point(y * scale_arrow_flow, x * scale_arrow_flow), 
                        cv::Point((y + e.best_v / 20) * scale_arrow_flow, (x + e.best_u / 20) * scale_arrow_flow), CV_RGB(0,0,255));
    #endif
    }

    cv::Mat flow_bgr;

    #if CV_MAJOR_VERSION == 2
        cv::cvtColor(flow_hsv, flow_bgr, CV_HSV2BGR);
    #elif CV_MAJOR_VERSION >= 3
        cv::cvtColor(flow_hsv, flow_bgr, cv::COLOR_HSV2BGR);
    #endif

    cv::convertScaleAbs(scores, scores, 10.0, 0);

    while (cv::waitKey(33) != 27) {
        cv::imshow(fl_window_name, flow_bgr);
        cv::imshow(best_pr_hires_window_name, best_project_hires_img);
        cv::imshow(fl_arrow_window_name, flow_arrow);
    }
}



template<class T> cv::Mat EventFile::projection_img (T *events, int scale, bool show_final, double min_t, 
                                                            double max_t) {

    int scale_img_x = RES_X * scale;
    int scale_img_y = RES_Y * scale;

    sll lt = FROM_SEC(min_t);
    sll rt = FROM_SEC(max_t);

    int cnt = 0;
    cv::Mat best_project_hires_img = cv::Mat::zeros(scale_img_x, scale_img_y, CV_8UC1);
    for (auto &e : *events) {
        if (e.noise) continue;

        if (max_t > min_t && max_t > 0) {
            if ((sll)e.timestamp < lt) continue;
            if ((sll)e.timestamp > rt) break;
        }

        //e.set_local_time(lt);
        //e.project_dn(0, 0);

        int x = e.pr_x * scale;
        int y = e.pr_y * scale;

        if (show_final) {
            x = e.fr_x * scale;
            y = e.fr_y * scale;
        }

        if ((x >= scale * (RES_X - 1)) || (x < 0) || (y >= scale * (RES_Y - 1)) || (y < 0))
            continue;

        x += scale / 2;
        y += scale / 2;

        cnt ++;

        int lx = std::max(x - scale / 2, 0), ly = std::max(y - scale / 2, 0);
        int rx = std::min(x + scale / 2, scale_img_x), ry = std::min(y + scale / 2, scale_img_y);
        for (int jx = lx; jx <= rx; ++jx) {
            for (int jy = ly; jy <= ry; ++jy) {
                if (best_project_hires_img.at<uchar>(jx, jy) < 255)
                    best_project_hires_img.at<uchar>(jx, jy) ++;
            }
        }
    }

    if (scale > 1) {
        cv::GaussianBlur(best_project_hires_img, best_project_hires_img, cv::Size(scale, scale), 0, 0);
    }

    double img_scale = 127.0 / EventFile::nonzero_average(best_project_hires_img);
    cv::convertScaleAbs(best_project_hires_img, best_project_hires_img, img_scale, 0);
    return best_project_hires_img;
}


template<class T> cv::Mat EventFile::projection_img_unopt (T *events, int scale) {

    int scale_img_x = RES_X * scale;
    int scale_img_y = RES_Y * scale;


    int cnt = 0;
    cv::Mat best_project_hires_img = cv::Mat::zeros(scale_img_x, scale_img_y, CV_8UC1);
    for (auto &e : *events) {
        if (e.noise) continue;

        int x = e.fr_x * scale;
        int y = e.fr_y * scale;

        if ((x >= scale * (RES_X - 1)) || (x < 0) || (y >= scale * (RES_Y - 1)) || (y < 0))
            continue;

        x += scale / 2;
        y += scale / 2;

        cnt ++;

        int lx = std::max(x - scale / 2, 0), ly = std::max(y - scale / 2, 0);
        int rx = std::min(x + scale / 2, scale_img_x), ry = std::min(y + scale / 2, scale_img_y);
        for (int jx = lx; jx <= rx; ++jx) {
            for (int jy = ly; jy <= ry; ++jy) {
                if (best_project_hires_img.at<uchar>(jx, jy) < 255)
                    best_project_hires_img.at<uchar>(jx, jy) ++;
            }
        }
    }

    if (scale > 1) {
        cv::GaussianBlur(best_project_hires_img, best_project_hires_img, cv::Size(scale, scale), 0, 0);
    }

    double img_scale = 127.0 / EventFile::nonzero_average(best_project_hires_img);
    cv::convertScaleAbs(best_project_hires_img, best_project_hires_img, img_scale, 0);
    return best_project_hires_img;
}


template<class T> cv::Mat EventFile::color_clusters_img (T *events, bool show_final) {
    double scale = 11;

    uint x_min = RES_X, y_min = RES_Y, x_max = 0, y_max = 0;
    for (auto &e : *events) {
        if (e.get_x() > x_max) x_max = e.get_x();
        if (e.get_y() > y_max) y_max = e.get_y();
        if (e.get_x() < x_min) x_min = e.get_x();
        if (e.get_y() < y_min) y_min = e.get_y();
    }

    x_max = std::min(x_max, (uint)RES_X);
    y_max = std::min(y_max, (uint)RES_Y);

    if ((x_min > x_max) || (y_min > y_max)) {
        return cv::Mat(0, 0, CV_8UC3, cv::Scalar(0, 0, 0));
    }

    int metric_wsizex = scale * (x_max - x_min);
    int metric_wsizey = scale * (y_max - y_min);
    int scale_img_x = metric_wsizex + scale;
    int scale_img_y = metric_wsizey + scale;

    int cluster_cnt = 6;

    cv::Mat project_img(scale_img_x, scale_img_y, CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat project_img_cnt = cv::Mat::zeros(scale_img_x, scale_img_y, CV_8UC1);
    cv::Mat project_img_avg(scale_img_x, scale_img_y, CV_8UC3, cv::Scalar(0, 0, 0));

    double x_shift = - double((x_max - x_min) / 2 + x_min) * double(scale) + double(metric_wsizex) / 2.0;
    double y_shift = - double((y_max - y_min) / 2 + y_min) * double(scale) + double(metric_wsizey) / 2.0;

    for (auto &e : *events) {
        if (e.noise) continue;
        if (e.cl == NULL) continue;

        int x = e.pr_x * scale + x_shift;
        int y = e.pr_y * scale + y_shift;

        if (show_final) {
            x = e.best_pr_x * scale + x_shift;
            y = e.best_pr_y * scale + y_shift;
        }

        if ((x >= metric_wsizex) || (x < 0) || (y >= metric_wsizey) || (y < 0))
            continue;
  
        float angle = 2 * 3.14 * double(e.cl->get_id() % cluster_cnt) / double(cluster_cnt);

        x += scale / 2;
        y += scale / 2;
        
        for (int jx = x - scale / 2; jx <= x + scale / 2; ++jx) {
            for (int jy = y - scale / 2; jy <= y + scale / 2; ++jy) {
                project_img.at<cv::Vec3f>(jx, jy)[0] += cos(angle);
                project_img.at<cv::Vec3f>(jx, jy)[1] += sin(angle);
                project_img_cnt.at<uchar>(jx, jy) ++;
            }
        }
    }

    for (int jx = 0; jx < scale_img_x; ++jx) {
        for (int jy = 0; jy < scale_img_y; ++jy) {
            if (project_img_cnt.at<uchar>(jx, jy) == 0) continue;

            float vx = (project_img.at<cv::Vec3f>(jx, jy)[0] / (float)project_img_cnt.at<uchar>(jx, jy));
            float vy = (project_img.at<cv::Vec3f>(jx, jy)[1] / (float)project_img_cnt.at<uchar>(jx, jy));

            double speed = hypot(vx, vy);
            double angle = 0;
            if (speed != 0)
                angle = (atan2(vy, vx) + 3.1416) * 180 / 3.1416;
            
            project_img_avg.at<cv::Vec3b>(jx, jy)[0] = angle / 2;
            project_img_avg.at<cv::Vec3b>(jx, jy)[1] = speed * 255;
            project_img_avg.at<cv::Vec3b>(jx, jy)[2] = 255;
        }
    }

    #if CV_MAJOR_VERSION == 2
        cv::cvtColor(project_img_avg, project_img_avg, CV_HSV2BGR);
    #elif CV_MAJOR_VERSION >= 3
        cv::cvtColor(project_img_avg, project_img_avg, cv::COLOR_HSV2BGR);
    #endif

    return project_img_avg;
}


template<class T> cv::Mat EventFile::color_time_img (T *events, int scale, bool show_final) {
    if (scale == 0) scale = 11;

    ull t_min = LLONG_MAX, t_max = 0;
    uint x_min = RES_X, y_min = RES_Y, x_max = 0, y_max = 0;
    for (auto &e : *events) {
        if (e.get_x() > x_max) x_max = e.get_x();
        if (e.get_y() > y_max) y_max = e.get_y();
        if (e.get_x() < x_min) x_min = e.get_x();
        if (e.get_y() < y_min) y_min = e.get_y();
    }

    for (auto &e : *events) {
        if (e.t < (sll)t_min) t_min = e.t;
        if (e.t > (sll)t_max) t_max = e.t;
    }

    x_max = std::min(x_max, (uint)RES_X);
    y_max = std::min(y_max, (uint)RES_Y);
    x_min = y_min = 0;
    x_max = RES_X;
    y_max = RES_Y;

    if ((x_min > x_max) || (y_min > y_max)) {
        return cv::Mat(0, 0, CV_8UC3, cv::Scalar(0, 0, 0));
    }

    int metric_wsizex = scale * (x_max - x_min);
    int metric_wsizey = scale * (y_max - y_min);
    int scale_img_x = metric_wsizex + scale;
    int scale_img_y = metric_wsizey + scale;

    cv::Mat project_img(scale_img_x, scale_img_y, CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat project_img_cnt = cv::Mat::zeros(scale_img_x, scale_img_y, CV_32FC1);
    cv::Mat project_img_avg(scale_img_x, scale_img_y, CV_8UC3, cv::Scalar(0, 0, 0));

    double x_shift = - double((x_max - x_min) / 2 + x_min) * double(scale) + double(metric_wsizex) / 2.0;
    double y_shift = - double((y_max - y_min) / 2 + y_min) * double(scale) + double(metric_wsizey) / 2.0;

    int ignored = 0, total = 0;
    for (auto &e : *events) {
        total ++;
        if (e.noise) continue;

        int x = e.pr_x * scale + x_shift;
        int y = e.pr_y * scale + y_shift;

        if (show_final) {
            x = e.fr_x * scale + x_shift;
            y = e.fr_y * scale + y_shift;
        }

        if ((x >= metric_wsizex) || (x < 0) || (y >= metric_wsizey) || (y < 0)) {
            ignored ++;
            continue;
        }

        float angle = 2 * 3.14 * (double(e.t - t_min) / double(t_max - t_min));

        x += scale / 2;
        y += scale / 2;
        
        for (int jx = x - scale / 2; jx <= x + scale / 2; ++jx) {
            for (int jy = y - scale / 2; jy <= y + scale / 2; ++jy) {
                project_img.at<cv::Vec3f>(jx, jy)[0] += cos(angle);
                project_img.at<cv::Vec3f>(jx, jy)[1] += sin(angle);
                project_img_cnt.at<float>(jx, jy) += 1;
            }
        }
    }
        
    for (int jx = 0; jx < scale_img_x; ++jx) {
        for (int jy = 0; jy < scale_img_y; ++jy) {
            if (project_img_cnt.at<float>(jx, jy) < 1) continue;

            float vx = (project_img.at<cv::Vec3f>(jx, jy)[0] / (float)project_img_cnt.at<float>(jx, jy));
            float vy = (project_img.at<cv::Vec3f>(jx, jy)[1] / (float)project_img_cnt.at<float>(jx, jy));

            double speed = hypot(vx, vy);
            double angle = 0;
            if (speed != 0)
                angle = (atan2(vy, vx) + 3.1416) * 180 / 3.1416;
            
            project_img_avg.at<cv::Vec3b>(jx, jy)[0] = angle / 2;
            project_img_avg.at<cv::Vec3b>(jx, jy)[1] = speed * 255;
            project_img_avg.at<cv::Vec3b>(jx, jy)[2] = 255;
        }
    }

    //std::cout << "Ignored / Total: " << ignored << " / " << total << std::endl;

    #if CV_MAJOR_VERSION == 2
        cv::cvtColor(project_img_avg, project_img_avg, CV_HSV2BGR);
    #elif CV_MAJOR_VERSION >= 3
        cv::cvtColor(project_img_avg, project_img_avg, cv::COLOR_HSV2BGR);
    #endif

    return project_img_avg;
}




#endif // EVENT_FILE_H
