#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <iostream>
#include <time.h>
#include "solver.hpp"
#include "common.hpp"

#define uchar unsigned char

#ifdef CUDA
#include <cuda_runtime.h>
#endif

#ifdef MPI
#include <mpi.h>
#endif

int main(int argc, char **argv) {
  int max_iter = 8000;
  int x = 1920;
  int y = 1080;
  int rank = 0;
  int num_procs = 1;

  double real_max = 1.5;
  double real_min = -2.0;
  double imag_max = 1.0;
  double imag_min = -1.0;
  
  // I/O
  bool output = false;
  std::string output_file = "output.ppm";

  #ifdef MPI 
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (MPI_SUCCESS != res) {
        fprintf(stderr, "MPI_Init failed\n");
        return 1;
    }
  #endif

  std::cout << "rank: " << rank << std::endl;

  int cur_arg = 1;
  int num_args = argc - 1;

  while (num_args > 0) {
    if (num_args == 1) {
      fprintf(stderr, "Missing argument value for %s\n", argv[cur_arg]);
      return 1;
    }
    if (strcmp(argv[cur_arg], "--x") == 0) {                 // x resolution
      x = atoi(argv[cur_arg + 1]);
    } else if (strcmp(argv[cur_arg], "--y") == 0) {          // y resolution
      y = atoi(argv[cur_arg + 1]);
    } else if (strcmp(argv[cur_arg], "--max_iter") == 0) {   // max iterations
      max_iter = atoi(argv[cur_arg + 1]);
    } else if (strcmp(argv[cur_arg], "--real_max") == 0) {   // real-dimension max
      real_max = atof(argv[cur_arg + 1]);
    } else if (strcmp(argv[cur_arg], "--real_min") == 0) {   // real-dimension min
      real_min = atof(argv[cur_arg + 1]);
    } else if (strcmp(argv[cur_arg], "--imag_max") == 0) {   // imaginary-dimension max
      imag_max = atof(argv[cur_arg + 1]);
    } else if (strcmp(argv[cur_arg], "--imag_min") == 0) {   // imaginary-dimension min
      imag_min = atof(argv[cur_arg + 1]);
    } else if (strcmp(argv[cur_arg], "--output") == 0) {     // output filec
      output = true;
      output_file = argv[cur_arg + 1];
    } else {
      fprintf(stderr, "Unknown argument %s\n", argv[cur_arg]);
      return 1;
    }
    cur_arg += 2;
    num_args -= 2;
  }
  std::cout << "x: " << x << std::endl;
  std::cout << "y: " << y << std::endl;
  std::cout << "max_iter: " << max_iter << std::endl;
  std::cout << "real_max: " << real_max << std::endl;
  std::cout << "real_min: " << real_min << std::endl;
  std::cout << "imag_max: " << imag_max << std::endl;
  std::cout << "imag_min: " << imag_min << std::endl;
  std::cout << "output: " << output << std::endl;
  std::cout << "output_file: " << output_file << std::endl;


  #ifdef CUDA
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device name: " << prop.name << std::endl;
  #endif

  uchar* colors = common::make_gradient(max_iter);
  uchar *grid = new uchar[x * y * 3];

  clock_t init_start = clock();
  init(grid, colors, x, y, max_iter, real_max, real_min, imag_max, imag_min, rank, num_procs);
  clock_t init_end = clock();
  double init_elapsed = (double)(init_end - init_start) / CLOCKS_PER_SEC;
  std::cout << "Initialization time: " << init_elapsed << std::endl;

  clock_t exec_start = clock();
  run();

  #ifdef CUDA
    cudaDeviceSynchronize();
  #endif

  clock_t exec_end = clock();
  double exec_elapsed = (double)(exec_end - exec_start) / CLOCKS_PER_SEC;
  std::cout << "Run time: " << exec_elapsed << std::endl;

  FILE *fptr;
  if (rank == 0 && output) {
    // add ppm extension
    std::string output_file_ppm = output_file + ".ppm";
    fptr = fopen(output_file_ppm.c_str(), "wb");
    if (fptr == NULL) {
      fprintf(stderr, "Error opening file %s\n", output_file_ppm.c_str());
      return 1;
    }
    transfer_data_to_host((uchar *)grid);
    fprintf(fptr, "P6\n%d %d\n255\n", x, y);
    for (int py = 0; py < y; py++)
        for (int px = 0; px < x; px++) 
          fwrite(&grid[py * x * 3 + px * 3], 1, 3, fptr);
  }
  

  clock_t free_start = clock();
  free_memory();
  clock_t free_end = clock();
  double free_elapsed = (double)(free_end - free_start) / CLOCKS_PER_SEC;
  std::cout << "Free time: " << free_elapsed << std::endl;

  if (rank == 0 && output) {
    fclose(fptr);
    delete[] grid;
    delete[] colors;
  }

  #ifdef MPI
    MPI_Finalize();
  #endif

  return 0;
}
