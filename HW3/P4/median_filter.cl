#include "median9.h"
int get_index(int w, int h, int x, int y);
//help function to handle index on boundry
int get_index(int w, int h, int x, int y){
    if(x < 0){
        x = 0;
    }
    if(x >= w){
        x = w - 1;
    }
    if(y >= h){
        y = h - 1;
    }
    if(y < 0){
        y = 0;
    }
    return y*w+x;
}

// 3x3 median filter
__kernel void
median_3x3(__global __read_only float *in_values,
           __global __write_only float *out_values,
           __local float *buffer,
           int w, int h,
           int buf_w, int buf_h,
           const int halo)
{
    // Note: It may be easier for you to implement median filtering
    // without using the local buffer, first, then adjust your code to
    // use such a buffer after you have that working.
    
    //get global location
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    //get local locatoin
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    //1-D index of local location
    const int index = ly * get_local_size(0) + lx;

    //get corner location of buffer

    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    //get location of buffer
    const int buf_x = lx + halo;
    const int buf_y = ly + halo;

    

    // Load into buffer (with 1-pixel halo).
    //
    // It may be helpful to consult HW3 Problem 5, and
    // https://github.com/harvard-cs205/OpenCL-examples/blob/master/load_halo.cl
    //
    // Note that globally out-of-bounds pixels should be replaced
    // with the nearest valid pixel's value.

    //code refered to load_halo.cl
    if (index < buf_w) {// From load_halo.cl
        for (int row = 0; row < buf_h; row++) {
            buffer[row * buf_w + index] = in_values[get_index(w, h, buf_corner_x + index, buf_corner_y + row)];
        }
    }

    // Compute 3x3 median for each pixel in core (non-halo) pixels
    //
    // We've given you median9.h, and included it above, so you can
    // use the median9() function.
    barrier(CLK_LOCAL_MEM_FENCE);
    
    
    // Each thread in the valid region (x < w, y < h) should write
    // back its 3x3 neighborhood median.
    if ((x < w) && (y < h)) {
        out_values[y*w+x] = median9(
                            buffer[get_index(buf_w, buf_h, buf_x-1, buf_y-1)],
                            buffer[get_index(buf_w, buf_h, buf_x, buf_y-1)],
                            buffer[get_index(buf_w, buf_h, buf_x+1, buf_y-1)],
                            buffer[get_index(buf_w, buf_h, buf_x-1, buf_y)],
                            buffer[get_index(buf_w, buf_h, buf_x, buf_y)],
                            buffer[get_index(buf_w, buf_h, buf_x+1, buf_y)],
                            buffer[get_index(buf_w, buf_h, buf_x-1, buf_y+1)],
                            buffer[get_index(buf_w, buf_h, buf_x, buf_y+1)],
                            buffer[get_index(buf_w, buf_h, buf_x+1, buf_y+1)]);
    }
}
