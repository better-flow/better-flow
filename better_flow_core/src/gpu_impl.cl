void kernel pr_4param(global const int* fr_x, global const int* fr_y, global float* nx,
                      global float* ny, global float* pr_x, global float* pr_y, global const int* t,
                      float dnx_, float dny_, float cx, float cy, float div, float crl) {
    int i = get_global_id(0);
    float2 r = (float2)(pr_x[i] - cx, pr_y[i] - cy);
    float cos_ = 0;
    float sin_ = sincos(crl, &cos_);
    float2 r_ = (float2)(cos_ * r.x - sin_ * r.y, sin_ * r.x + cos_ * r.y);
    float2 dn = - r_ * div + (r_ - r);
    nx[i] += dn.x + dnx_;
    ny[i] += dn.y + dny_;
    float kx = nx[i] / 127;
    float ky = ny[i] / 127;
    
    float x0 = fr_x[i];
    float y0 = fr_y[i];
    float t0 = t[i];

    pr_x[i] = x0 - t0 * kx / 10000.0;
    pr_y[i] = y0 - t0 * ky / 10000.0;
}


void kernel time_img(global float* pr_y, global float* pr_x, global float* t, 
                     global float* timg, global float* cimg, global float* img_add,
                     int w, int h, int scale, int x_sh, int y_sh) {
    int i = get_global_id(0);
    int w_real = w + scale;
    int h_real = h + scale;

    int h_scale = scale / 2;

    int x = pr_x[i] * scale + x_sh;
    int y = pr_y[i] * scale + y_sh;

    if ((x >= h + h_scale) || (x < h_scale) || (y >= w + h_scale) || (y < h_scale))
        return;
  
    for (int jy = y - h_scale; jy <= y + h_scale; ++jy) {
        for (int jx = x - h_scale; jx <= x + h_scale; ++jx) {
            img_add[jy * h_real + jx] += t[i];
            cimg[jy * h_real + jx] += 1;
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int jy = y - h_scale; jy <= y + h_scale; ++jy) {
        for (int jx = x - h_scale; jx <= x + h_scale; ++jx) {
            timg[jy * h_real + jx] = img_add[jy * h_real + jx] / cimg[jy * h_real + jx];
        }
    }
}


void kernel sharr(global float* img, global float* grad_x, global float* grad_y, int w, int h) {
    int i = get_global_id(0);
    if (i >= w * (h - 2)) return;
    int id_0 = i + 1;
    int id_1 = i + w + 1;
    int id_2 = i + 2 * w + 1;

    //int sharr_x [] = {3, 0, -3, 10, 0, -10, 3, 0, -3};
    //int sharr_y [] = {3, 10, 3, 0, 0, 0, -3, -10, -3};

    float th = 0.00001;
    if ((img[id_0 - 1] < th) || (img[id_0] < th) || (img[id_0 + 1] < th) ||
        (img[id_1 - 1] < th) || (img[id_1] < th) || (img[id_1 + 1] < th) ||
        (img[id_2 - 1] < th) || (img[id_2] < th) || (img[id_2 + 1] < th))
        return;

    float dx = 0, dy = 0;
    dy +=  3 * img[id_0 - 1] +  0 * img[id_0] -  3 * img[id_0 + 1];
    dy += 10 * img[id_1 - 1] +  0 * img[id_1] - 10 * img[id_1 + 1];
    dy +=  3 * img[id_2 - 1] +  0 * img[id_2] -  3 * img[id_2 + 1];

    dx +=  3 * img[id_0 - 1] + 10 * img[id_0] +  3 * img[id_0 + 1];
    dx +=  0 * img[id_1 - 1] +  0 * img[id_1] -  0 * img[id_1 + 1];
    dx += -3 * img[id_2 - 1] - 10 * img[id_2] -  3 * img[id_2 + 1];

    grad_x[id_1] = dx;
    grad_y[id_1] = dy;
}


void kernel model_helper(global float* img, global float* grad_x, global float* grad_y, 
                         global float* rots, global float* divs,
                         float cx, float cy, int w, int h) {
    int i = get_global_id(0);
    int id_0 = i + 1;
    int id_1 = i + w + 1;
    int id_2 = i + 2 * w + 1;

    grad_x[id_1] = 0;
    grad_y[id_1] = 0;
    rots[id_1] = 0;
    divs[id_1] = 0;

    if (i >= w * (h - 2)) return;

    //int sharr_x [] = {3, 0, -3, 10, 0, -10, 3, 0, -3};
    //int sharr_y [] = {3, 10, 3, 0, 0, 0, -3, -10, -3};

    
    float th = 0.00001;
    if ((img[id_0 - 1] < th) || (img[id_0] < th) || (img[id_0 + 1] < th) ||
        (img[id_1 - 1] < th) || (img[id_1] < th) || (img[id_1 + 1] < th) ||
        (img[id_2 - 1] < th) || (img[id_2] < th) || (img[id_2 + 1] < th))
        return;
    

    float dx = 0, dy = 0;
    dy +=  3 * img[id_0 - 1] +  0 * img[id_0] -  3 * img[id_0 + 1];
    dy += 10 * img[id_1 - 1] +  0 * img[id_1] - 10 * img[id_1 + 1];
    dy +=  3 * img[id_2 - 1] +  0 * img[id_2] -  3 * img[id_2 + 1];

    dx +=  3 * img[id_0 - 1] + 10 * img[id_0] +  3 * img[id_0 + 1];
    dx +=  0 * img[id_1 - 1] +  0 * img[id_1] -  0 * img[id_1 + 1];
    dx += -3 * img[id_2 - 1] - 10 * img[id_2] -  3 * img[id_2 + 1];

    grad_x[id_1] = dx;
    grad_y[id_1] = dy;

    int y = id_1 % w;
    int x = id_1 / w;
    float2 r = (float2)(x - cx, y - cy);
    float2 g = (float2)(dx, dy);

    rots[id_1] = r.x * g.y - r.y * g.x;
    divs[id_1] = dot(r, g);
}