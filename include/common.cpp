#include "common.hpp"
#include <cmath>

// double common::interpolate(double v0, double v1, double t) {
//   return (1 - t) * v0 + t * v1;
// }

uchar* common::make_gradient(int size) {
  uchar* gradient = new uchar[3 * (size + 1)];
  for (int i = 0; i < 3*(size + 1); i += 3) {
    if (i >= 3*size) {
      gradient[i] = 0;
      gradient[i + 1] = 0;
      gradient[i + 2] = 0;
      continue;
    }

    double j = i == 0 ? 3.0 : 3.0 * (log(i) / log(size - 1.0));

    if (j < 1) {
      gradient[i] = 255 * j;
      gradient[i + 1] = 0;
      gradient[i + 2] = 0;
    } else if (j < 2) {
      gradient[i] = 255;
      gradient[i + 1] = 255 * (j - 1);
      gradient[i + 2] = 0;
    } else {
      gradient[i] = 255;
      gradient[i + 1] = 255;
      gradient[i + 2] = 255 * (j - 2);
    }
  }
  
  return gradient;
}