#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <ctime>
#include <new>
#include <cassert>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


typedef long int lint;
typedef long long int llint;
typedef unsigned int uint;
typedef unsigned long int ulong;
typedef unsigned long long int ull;



// Input parameters
int resolution_x = 240;
int resolution_y = 180;

bool verbose = true;

std::vector<uint> x_arr;
std::vector<uint> y_arr;
std::vector<ull>  t_arr;
std::vector<bool> p_arr;

// Range of events
ull min_actual_time = 0;
ull max_actual_time = 0;

// Event slice we are looking at (in ms * 10)
int min_slice_time = 0; // 0.0 sec
int width_slice_time = 1000; // 100 ms


void read_events (std::string fname, double llimit, double hlimit) {
    std::cout << "Reading from file... (" << fname << ")" << std::endl << std::flush;

    std::ifstream event_file(fname, std::ifstream::in);

    ull cnt = 0;
    double t = 0;
    uint x = 0, y = 0;
    bool p = false;

    double t_0 = 0; // the earliest timestamp in the file
    clock_t begin = std::clock();


    event_file >> t_0 >> x >> y >> p;
    while (event_file >> t >> x >> y >> p) {
        if (t - t_0 > llimit)
            break;
    }

    while (event_file >> t >> x >> y >> p) {
        t -= t_0;
        if (t  > hlimit)
            break;

        x_arr.push_back(x);
        y_arr.push_back(y);
        t_arr.push_back((t - llimit) * 1000000000);
        p_arr.push_back(p);
        cnt ++;
    }
    clock_t end = std::clock();

    event_file.close();

    if (cnt == 0) {
        std::cout << "Read " << cnt << " events, finished" << std::endl << std::endl << std::flush;
        return;
    }

    min_actual_time = t_arr[0];
    max_actual_time = t_arr[t_arr.size() - 1];

    std::cout << "Read " << cnt << " events, finished" << std::endl << std::flush;
    std::cout << "Elapsed: " << double(end - begin) / CLOCKS_PER_SEC << " sec." << std::endl << std::flush;
    std::cout << "Time diff: " << t_arr[t_arr.size() - 1] - t_arr[0] << std::endl << std::flush;
    std::cout << "Time diff: " << (long double)(t_arr[t_arr.size() - 1] - t_arr[0]) / 1000000000.0 
              << " sec." << std::endl << std::endl << std::flush;
}

// Project grayscale
cv::Mat project_events (double nx, double ny, double nz) {
    int scale = 3;
    
    cv::Mat project_img = cv::Mat::zeros(resolution_x * scale + scale, resolution_y * scale + scale, CV_8UC1);
    if (nz == 0) return project_img;


    double t_divider = 1;
    double xy_len = hypot(nx, ny);
    double speed = xy_len / (nz / (1000000000/(t_divider * 10000)));
    double u = (xy_len == 0) ? 0 : speed * nx / xy_len;
    double v = (xy_len == 0) ? 0 : speed * ny / xy_len;

    if (verbose)
        std::cout << "Flow speed: " << speed << "; (u, v): (" << u << ", " << v 
                  << ")  \t" << nx << " " << ny << " " << nz;

    // Choose slice events:
    ull i = 0;
    for (; i < x_arr.size(); ++i) {
        if (ull(min_slice_time) * 100000 < t_arr[i])
            break;
    }

    clock_t begin = std::clock();
    double kx = nx / nz;
    double ky = ny / nz;

   
    for (; i < x_arr.size(); ++i) {
        if ((ull(min_slice_time) + ull(width_slice_time)) * 100000 < t_arr[i])
            break;

        int x = scale * (x_arr[i] - double((t_arr[i] - ull(min_slice_time) * 100000) / t_divider) / 10000 * kx);
        int y = scale * (y_arr[i] - double((t_arr[i] - ull(min_slice_time) * 100000) / t_divider) / 10000 * ky);

        if ((x >= scale * resolution_x) || (x < 0) || (y >= scale * resolution_y) || (y < 0))
            continue;

        for (int jx = x; jx < x + scale; ++jx) {
            for (int jy = y; jy < y + scale; ++jy) {
                if (project_img.at<uchar>(jx, jy) < 255)
                    project_img.at<uchar>(jx, jy) ++;
            }
        }
    }

    if (scale > 1) {
        int k_size = (scale % 2 == 0) ? scale + 1 : scale;
        cv::GaussianBlur(project_img, project_img, cv::Size(k_size, k_size), 0, 0);
    }

    clock_t end = std::clock();

    if (verbose)
        std::cout << "\t Elapsed: " << double(end - begin) / CLOCKS_PER_SEC << " sec.\n";

    return project_img;
}


// Project color
cv::Mat project_events_color (double nx, double ny, double nz) {
    int scale = 3;
    
    cv::Mat project_img(resolution_x * scale + scale, resolution_y * scale + scale, CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat project_img_cnt = cv::Mat::zeros(resolution_x * scale + scale, resolution_y * scale + scale, CV_8UC1);
    if (nz == 0) return project_img;

    double t_divider = 1;
    double xy_len = hypot(nx, ny);
    double speed = xy_len / (nz / (1000000000/(t_divider * 10000)));
    double u = (xy_len == 0) ? 0 : speed * nx / xy_len;
    double v = (xy_len == 0) ? 0 : speed * ny / xy_len;

    if (verbose)
        std::cout << "Flow speed (color): " << speed << "; (u, v): (" << u << ", " << v 
                  << ")  \t" << nx << " " << ny << " " << nz;

    // Choose slice events:
    ull i = 0;
    for (; i < x_arr.size(); ++i) {
        if (ull(min_slice_time) * 100000 < t_arr[i])
            break;
    }

    clock_t begin = std::clock();
    double kx = nx / nz;
    double ky = ny / nz;

    ull t_min = t_arr[i];
    ull t_max = t_min;
    for (ull j = i; j < t_arr.size(); ++j) {
        if ((ull(min_slice_time) + ull(width_slice_time)) * 100000 < t_arr[j])
            break;

        if (t_arr[j] > t_max) t_max = t_arr[j];
    }

    for (; i < x_arr.size(); ++i) {
        if ((ull(min_slice_time) + ull(width_slice_time)) * 100000 < t_arr[i])
            break;

        int x = scale * (x_arr[i] - double((t_arr[i] - ull(min_slice_time) * 100000) / t_divider) / 10000 * kx);
        int y = scale * (y_arr[i] - double((t_arr[i] - ull(min_slice_time) * 100000) / t_divider) / 10000 * ky);

        if ((x >= scale * resolution_x) || (x < 0) || (y >= scale * resolution_y) || (y < 0))
            continue;

        float angle = 2 * 3.14 * (double(t_arr[i] - t_min) / double(t_max - t_min));
        for (int jx = x; jx < x + scale; ++jx) {
            for (int jy = y; jy < y + scale; ++jy) {
                project_img.at<cv::Vec3f>(jx, jy)[0] += cos(angle);
                project_img.at<cv::Vec3f>(jx, jy)[1] += sin(angle);
                project_img_cnt.at<uchar>(jx, jy) ++;
            }
        }
    }
        
    cv::Mat project_img_avg(resolution_x * scale + scale, resolution_y * scale + scale, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int jx = 0; jx < resolution_x * scale + scale; ++jx) {
        for (int jy = 0; jy < resolution_y * scale + scale; ++jy) {
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

    cv::cvtColor(project_img_avg, project_img_avg, CV_HSV2BGR);

    //if (scale > 1) {
    //    int k_size = (scale % 2 == 0) ? scale + 1 : scale;
    //    cv::GaussianBlur(project_img, project_img, cv::Size(k_size, k_size), 0, 0);
    //}

    clock_t end = std::clock();

    if (verbose)
        std::cout << "\t Elapsed: " << double(end - begin) / CLOCKS_PER_SEC << " sec.\n";

    //std::cout << "\t" <<  kt << " " << t_max << " " << t_min << " " << double(t_max - t_min) << "\n";
    return project_img_avg;
}




int value_x = 127, value_y = 127, fine = 500;
cv::Mat project_img;

void button_cb(int state, void* userdata) {
    double nx = double(value_x - 127) / double(fine + 1);
    double ny = double(value_y - 127) / double(fine + 1);
    project_img = project_events(nx, ny, 127);
}


double nonzero_average (cv::Mat img) {
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


ull do_hist(cv::Mat img, std::string name, uint percentile = 90) {
    assert(percentile < 100);

    double nz_avg = nonzero_average(img);

    /// Establish the number of bins
    int histSize = 256;

    /// Set the ranges
    float range[] = { 0, 256 } ;
    const float* histRange = { range };

    bool uniform = true; bool accumulate = false;

    cv::Mat e_hist;

    /// Compute the histograms:
    cv::calcHist(&img, 1, 0, cv::Mat(), e_hist, 1, &histSize, &histRange, uniform, accumulate);
    e_hist.at<float>(0) = 0;

    // compute percentiles:
    ull total_pixels = 0;
    for (int i = 0; i < histSize; ++i)
        total_pixels += e_hist.at<float>(i);

    int left_prc = 0;
    ull small_cnt = 0;
    for (; left_prc < histSize; ++left_prc) {
        small_cnt += e_hist.at<float>(left_prc);
        if (small_cnt > (float(100 - percentile) / 100.0) * (total_pixels - small_cnt))
            break;
    }

    int rght_prc = histSize - 1;
    ull large_cnt = 0;
    for (; rght_prc >= 0; rght_prc--) {
        large_cnt += e_hist.at<float>(rght_prc);
        if (large_cnt > (float(100 - percentile) / 100.0) * (total_pixels - large_cnt))
            break;
    }

    // Draw the histograms for B, G and R
    int hist_w = 512; int hist_h = 800;
    int bin_w = cvRound((double)hist_w / histSize);

    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0,0,0));

    /// Normalize the result to [ 0, histImage.rows ]
    
    for (int i = 0; i < histSize; ++i)
        e_hist.at<float>(i) = std::log(1 + e_hist.at<float>(i));
    normalize(e_hist, e_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

    /// Draw
    for(int i = 1; i < histSize; ++i) {
        cv::line(histImage, cv::Point(bin_w*(i-1), hist_h - cvRound(e_hist.at<float>(i-1))),
                 cv::Point(bin_w*(i), hist_h - cvRound(e_hist.at<float>(i))),
                 cv::Scalar(0, 0, 255), 2, 8, 0);
    }
    
    cv::line(histImage, cv::Point(bin_w * left_prc, 0), cv::Point(bin_w * left_prc, hist_h),
                 cv::Scalar(0, 255, 0), 2, 8, 0);
    cv::line(histImage, cv::Point(bin_w * rght_prc, 0), cv::Point(bin_w * rght_prc, hist_h),
                 cv::Scalar(0, 255, 0), 2, 8, 0);

    /// Display
    cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
    cv::displayStatusBar(name, "(" + std::to_string(small_cnt) + " " + std::to_string(total_pixels) + 
                               " " + std::to_string(large_cnt) + ") avg: " + std::to_string(nz_avg));
    cv::imshow(name, histImage);

    return rght_prc;
}


cv::Mat do_sobel(cv::Mat img, bool show_img = true) {
    clock_t begin = std::clock();

    cv::Mat Gx, Gy, Gxy, Lap;
    cv::Sobel(img, Gx, CV_32F, 1, 0, 3);
    cv::Sobel(img, Gy, CV_32F, 0, 1, 3);
    cv::magnitude(Gx, Gy, Gxy);
    //cv::Laplacian(img, Lap, CV_16S, 5);

    //cv::normalize(cv::abs(Gx), Gx, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    //cv::normalize(cv::abs(Gy), Gy, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    cv::convertScaleAbs(Gx, Gx, 4.0, 0);
    cv::convertScaleAbs(Gy, Gy, 4.0, 0);
    cv::convertScaleAbs(Gxy, Gxy, 4.0, 0);
    //cv::convertScaleAbs(Lap, Lap);

    clock_t end = std::clock();

    if (verbose)
        std::cout << "\t\t Sobel elapsed: " << double(end - begin) / CLOCKS_PER_SEC << " sec.\n";

    if (!show_img) return Gxy;

    cv::namedWindow("sobel x", cv::WINDOW_NORMAL);
    cv::namedWindow("sobel y", cv::WINDOW_NORMAL);
    cv::namedWindow("sobel xy", cv::WINDOW_NORMAL);
    //cv::namedWindow("laplacian", cv::WINDOW_NORMAL);
    cv::imshow("sobel x", Gx);
    cv::imshow("sobel y", Gy);
    cv::imshow("sobel xy", Gxy);
    //cv::imshow("laplacian", Lap);

    //do_hist(Gx, "Sobel X hist");
    //do_hist(Gy, "Sobel Y hist");
    //do_hist(Gxy, "Sobel XY hist");
    //do_hist(Lap, "Laplacian hist");

    return Gxy;
}


void do_fft(cv::Mat img) {
    cv::namedWindow("fft Re", cv::WINDOW_NORMAL);
    cv::namedWindow("fft Im", cv::WINDOW_NORMAL);
    cv::namedWindow("fft Abs", cv::WINDOW_NORMAL);

    cv::Mat padded;
    int m = cv::getOptimalDFTSize(img.rows);
    int n = cv::getOptimalDFTSize(img.cols); // on the border add zero values
    cv::copyMakeBorder(img, padded, 0, m - img.rows, 0, n - img.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

    cv::dft(complexI, complexI);            // this way the result may fit in the source matrix

    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    cv::Mat magI;
    cv::split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    cv::magnitude(planes[0], planes[1], magI);         // planes[0] = magnitude


    cv::Mat logRe, logIm;
    logRe = cv::abs(planes[0]) + cv::Scalar::all(1); 
    cv::log(logRe, logRe);
    
    logIm = cv::abs(planes[1]) + cv::Scalar::all(1); 
    cv::log(logIm, logIm);

    magI += cv::Scalar::all(1);                    // switch to logarithmic scale
    cv::log(magI, magI);

    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));

    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols/2;
    int cy = magI.rows/2;

    cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));  // Top-Right
    cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));  // Bottom-Left
    cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

    /*
    cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                       // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
    */

    cv::normalize(magI,  magI,  0, 1, cv::NORM_MINMAX);                                                 
    cv::normalize(logRe, logRe, 0, 1, cv::NORM_MINMAX);
    cv::normalize(logIm, logIm, 0, 1, cv::NORM_MINMAX);

    cv::imshow("fft Re",  logRe);
    cv::imshow("fft Im",  logIm);
    cv::imshow("fft Abs", magI);
}


void generate_metric_plot () {
    double step = 0.001;
    double range = 0.1;
    verbose = false;

    double nx = -range;
    double ny = -range;

    for (; nx < range; ) {
        for (; ny < range; ) {
            project_img = project_events(nx, ny, 127);
            cv::Mat sobel_xy = do_sobel(project_img, false);
            
            //double nz_avg_primary = nonzero_average(project_img);
            double nz_avg_sobelxy = nonzero_average(sobel_xy);

            //std::cout << nx << ", " << ny << ", " << nz_avg_primary << ", " << nz_avg_sobelxy << "\n";
            //std::cout << nz_avg_primary << ", ";
            std::cout << nz_avg_sobelxy << ", ";

            ny += step;
        }
        std::cout << "\n";
        nx += step;
        ny = -range;
    }
}


double score (double nx, double ny) {
    project_img = project_events(nx, ny, 127);
    return nonzero_average(project_img);
}


double last_score = 0, dscore = 0;
double dnx = 0.1, dny = 0.1, dn_th = 0.001;
double compute_new_nx(double nx_, double ny) {
    double nx = nx_ + dnx;
    double new_score = score(nx, ny);

    dscore = new_score - last_score;
    last_score = new_score;
    
    if (dscore < 0) {    
        dnx = - dnx / 2.0;
    }

    return nx;
}


double compute_new_ny(double nx, double ny_) {
    double ny = ny_ + dny;
    double new_score = score(nx, ny);

    dscore = new_score - last_score;
    last_score = new_score;
    
    if (dscore < 0) {    
        dny = - dny / 2.0;
    }

    return ny;
}


double nx_grad = 0, ny_grad = 0;
void gradient_descent () {
    std::string window_name = "Projected";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);

    clock_t begin = std::clock();

    double nx = 0, ny = 0;
    last_score = score(nx, ny);
    // First x, then y, then both
    
    while (fabs(dnx) > dn_th) {
        nx = compute_new_nx(nx, ny);
        cv::normalize(project_img, project_img, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::imshow(window_name, project_img);
        cv::displayStatusBar(window_name, "primary avg: " + std::to_string(last_score));
        //cv::waitKey(0);
    }

    while (fabs(dny) > dn_th) {
        ny = compute_new_ny(nx, ny);
        cv::normalize(project_img, project_img, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::imshow(window_name, project_img);
        cv::displayStatusBar(window_name, "primary avg: " + std::to_string(last_score));
        //cv::waitKey(0);
    }

    dn_th /= 10;
    while (hypot(dnx, dny) > dn_th) {
        nx = compute_new_nx(nx, ny);
        ny = compute_new_ny(nx, ny);
    }

    clock_t end = std::clock();

    if (verbose)
        std::cout << "\t\t Minimization elapsed: " << double(end - begin) / CLOCKS_PER_SEC << " sec.\n";

    /*
    while (cv::waitKey(33) != 27) {
        cv::normalize(project_img, project_img, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::imshow(window_name, project_img);
        cv::displayStatusBar(window_name, "primary avg: " + std::to_string(last_score));
    }
    */

    nx_grad = nx;
    ny_grad = ny;
}


void flow_multitilt () {
    std::string window_name = "Projected";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    std::string window_name_color = "Projected Color";
    cv::namedWindow(window_name_color, cv::WINDOW_NORMAL);

    cv::createTrackbar("x tilt", window_name, &value_x, 255, NULL);
    cv::createTrackbar("y tilt", window_name, &value_y, 255, NULL);
    cv::createTrackbar("fine/coarse", window_name, &fine, 1000, NULL);
    cv::createTrackbar("start time (ms * 10)", window_name, &min_slice_time, max_actual_time / 100000, NULL);
    cv::createTrackbar("t width (ms * 10)", window_name, &width_slice_time, 1000, NULL);

    #if CV_MAJOR_VERSION == 3
        cv::createButton("Recompute", button_cb, NULL, CV_PUSH_BUTTON);
    #endif

    cv::setTrackbarPos("x tilt", window_name, (int)(nx_grad * (fine + 1) + 127) );
    cv::setTrackbarPos("y tilt", window_name, (int)(ny_grad * (fine + 1) + 127) );
    
    double nx = double(value_x - 127) / double(fine + 1);
    double ny = double(value_y - 127) / double(fine + 1);
    project_img = project_events(nx, ny, 127);

    while (cv::waitKey(33) != 27) {
        nx = double(value_x - 127) / double(fine + 1);
        ny = double(value_y - 127) / double(fine + 1);
        project_img = project_events(nx, ny, 127);

        //cv::Mat sobel_xy = do_sobel(project_img);
        //do_fft(project_img);
        //int prc = do_hist(project_img, "Project Hist");

        //double nz_avg_sobelxy = nonzero_average(sobel_xy);

        //cv::normalize(project_img, project_img, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        double img_scale = 127.0 / nonzero_average(project_img);
        cv::convertScaleAbs(project_img, project_img, img_scale, 0);

        cv::Mat project_img_color = project_events_color(nx, ny, 127);
        cv::imshow(window_name_color, project_img_color);
        cv::imshow(window_name, project_img);

        #if CV_MAJOR_VERSION == 3
            double nz_avg_primary = nonzero_average(project_img);
            cv::displayStatusBar(window_name, "primary avg: " + std::to_string(nz_avg_primary));// + 
                                          //" sobelxy avg: " + std::to_string(nz_avg_sobelxy));
        #endif
    }
}


int main(int argc, char *argv[]) {
    //read_events("../tools/events_gen.txt", 0.1, 0.2);
    //read_events("../datasets/events_0012.txt", 0.5, 1.0);
    //read_events("../datasets/shapes.txt", 24.28, 30);
    //read_events("../datasets/two_objects_opposite.txt", 5.6, 8);
    //read_events("../datasets/two_objects_perpendicular.txt", 4.5, 8);
    //read_events("../datasets/two_objects_same_direc.txt", 5.8, 7);
    //read_events("../datasets/2_objs_opp_spiral.txt", 5, 7);
    //read_events("../datasets/2_objs_opp_checkerboard.txt", 5, 6.5);
    //read_events("../datasets/events_6dof_trimmed_short.txt", 0, 5);

    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <file name> <start_time> <end_time>\n";
        return 1;
    }

    read_events(argv[1], atof(argv[2]), atof(argv[3]));


    gradient_descent();
    flow_multitilt();
    //generate_metric_plot();

    return 0;
}
