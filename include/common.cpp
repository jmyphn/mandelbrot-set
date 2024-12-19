#include "common.hpp"
#include <cmath>

double common::interpolate(double v0, double v1, double t) {
  return (1 - t) * v0 + t * v1;
}

common::RGB* common::make_gradient(int size) {
  RGB *gradient = new RGB[size + 1];
  for (int i = 0; i < size + 1; i++) {
    if (i >= size) {
      gradient[i] = RGB(0, 0, 0);
      continue;
    }

    double j = i == 0 ? 3.0 : 3.0 * (log(i) / log(size - 1.0));

    if (j < 1) gradient[i] = RGB(255 * j, 0, 0);
    else if (j < 2) gradient[i] = RGB((255), 255*(j - 1), 0);
    else gradient[i] = RGB(255, 255, 255 * (j - 2));
  }
  
  return gradient;
}