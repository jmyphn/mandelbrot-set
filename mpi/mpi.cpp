#include <iostream>
#include <cstring>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#include "../include/solver.hpp"

#define uchar unsigned char

int rank, num_procs;
int X, Y, max_iter;
double real_max, real_min, imag_max, imag_min;
RGB *colors;

uchar *grid;

void init(uchar* grid_, RGB *colors_, int X_, int Y_, int max_iter_, double real_max_, double real_min_, double imag_max_, double imag_min_, int rank_, int num_procs_) {
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
  if (rank == 0) {
    // parent
    MPI_Status status;
    int size = Y / (num_procs - 1);
    RGB* recv = new RGB[size * X];

    for (int i = 0; i < num_procs - 1; i++) {
      MPI_Recv(recv, sizeof(RGB) * size * X, MPI_CHAR, MPI_ANY_SOURCE, 1, MPI_COMM_WORD, &status);
      int source = status.MPI_SOURCE - 1;
      for (int px = 0; px < size; px++) {
        for (int py = 0; py < X; py++) {
          RGB color = recv[py * size + px];
          grid[(source * size + px) * X * 3 + py * 3] = color.r;
          grid[(source * size + px) * X * 3 + py * 3 + 1] = color.g;
          grid[(source * size + px) * X * 3 + py * 3 + 2] = color.b;
        }
      }
    }

  } else {
    // child
    int size = Y / (num_procs - 1);
    RGB *send = new RGB[size * X];
    for (int py = 0; py < size; py++) {
      for (int px = 0; px < X; px++) {
        int j = px * size + py;
        send[px * size + py] = find_mandelbrot(px, ((rank - 1) * size) + py);
      }
    }
    MPI_Send(buf, sizeof(Color) * size * X, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
    delete[] send;
  }
}

void free_memory() {
  return;
}

void transfer_data_to_host(double *grid) {
  return;
}