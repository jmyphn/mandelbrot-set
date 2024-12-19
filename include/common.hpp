#pragma once
#define uchar unsigned char

namespace common {
  class RGB {
    public:
      uchar r, g, b;
      RGB() : r(0), g(0), b(0) {};
      RGB(uchar _r, uchar _g, uchar _b): 
          r(_r), g(_g), b(_b) {};
  };

  double interpolate(double v0, double v1, double t);

  RGB* make_gradient(int size);
}
