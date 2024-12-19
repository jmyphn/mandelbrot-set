#pragma once
#include "common.hpp"

void init(unsigned char* grid_, unsigned char *colors_, int X_, int Y_, int max_iter_, double real_max_, double real_min_, double imag_max_, double imag_min_, int rank_, int num_procs_);

void run();

void free_memory();

void transfer_data_to_host(unsigned char *grid_);

unsigned char* find_mandelbrot(int px, int py);