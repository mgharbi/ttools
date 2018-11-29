#pragma once

#include "Halide.h"

using Halide::Expr;

Expr sigmoid(Expr x) {
  // Stable sigmoid
  return select(x>0,
                1.0f / (1.0f + exp(-x)),
                exp(x) / (1.0f + exp(x)));
}

Expr d_sigmoid(Expr x) {
  // Stable sigmoid
  Expr sx = sigmoid(x);
  return sx*(1.0f - sx);
}

Expr diff_abs(Expr x, float eps=1e-4) {
  return sqrt(x*x + eps*eps);
}
