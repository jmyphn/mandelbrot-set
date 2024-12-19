#include <iostream>
#include <complex>
#include <cstring>
#include <stdlib.h>
#include <math.h>
#include "../include/common.hpp"
#include "../include/solver.hpp"

#define RGB common::RGB

int X, Y, max_iter, rank, num_procs;

double real_max, real_min, imag_max, imag_min;

unsigned char *grid;
RGB *colors;

void init(unsigned char *grid_, RGB *colors_, int X_, int Y_, int max_iter_, double real_max_, double real_min_, double imag_max_, double imag_min_, int rank_, int num_procs_) {
    grid = grid_;
    colors = colors_;
    X = X_;
    Y = Y_;
    max_iter = max_iter_;
    real_max = real_max_;
    real_min = real_min_;
    imag_max = imag_max_;
    imag_min = imag_min_;
    rank = rank_;
    num_procs = num_procs_;
}

void run() {
    std::cout << "In run" << std::endl;
    for (int py = 0; py < Y; py++) {
        for (int px = 0; px < X; px++) {
            RGB color = find_mandelbrot(px, py);
            grid[py * X * 3 + px * 3] = color.r;
            grid[py * X * 3 + px * 3 + 1] = color.g;
            grid[py * X * 3 + px * 3 + 2] = color.b;
        }
    }
    std::cout << "Exiting run" << std::endl;
}

RGB find_mandelbrot(int px, int py) {
    double b = 0; // complex (c)
    double a = 0;

    double b0 = real_min + (px * ((real_max - real_min)/(X*1.0))); // complex scale of Px
    double a0 = imag_min + (py * ((imag_max - imag_min)/(Y*1.0))); // complex scale of Py

    double i = 0;
    double b2 = 0;
    double a2 = 0;

    while(b2 + a2 <= 20 && i < max_iter){
        a = 2*b*a + a0;
        b = b2 - a2 + b0;
        b2 = b*b;
        a2 = a*a;
        i++;
    }
    if(i < max_iter){
        double log_zn = log(b*b + a*a) / 2.0;
        double nu = log(log_zn / log(2.0))/log(2.0);
        i += 1.0 - nu;
    }
    RGB color1 = colors[(int)i];
    RGB color2 = colors[(int) i + 1 > max_iter ? (int) i : (int) i + 1];

    double t = i - (int) i;
    RGB color = RGB(
        common::interpolate(color1.r, color2.r, t),
        common::interpolate(color1.g, color2.g, t),
        common::interpolate(color1.b, color2.b, t)
    );
    return color;
}

void free_memory() {
  return;
}

void transfer_data_to_host(unsigned char *grid) {
  return;
}