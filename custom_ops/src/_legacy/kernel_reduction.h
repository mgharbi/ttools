#pragma once

#include <map>
#include <string>
#include <Halide.h>

#include "helpers.h"

using namespace Halide;

Var x("x"), y("y"), 
    dx("dx"), dy("dy"),
    ci("ci"), n("n");

template <typename InputBuffer>
std::map<std::string, Func> kernel_reduction(
        const InputBuffer &radiance, const InputBuffer &kernel
        ) {
    Func f_radiance("f_radiance");
    f_radiance(x, y, ci, n) = Halide::BoundaryConditions::constant_exterior(
        radiance, 0.0f)( x, y, ci, n);
    Func f_kernel("f_kernel");
    f_kernel(x, y, dx, dy, n) = Halide::BoundaryConditions::constant_exterior(
        kernel, 0.0f)(x, y, dx, dy, n);

    Expr kw = kernel.dim(2).extent();
    Expr kh = kernel.dim(3).extent();
    RDom r(0, kw, 0, kh);
    Func f_output("f_output");
    f_output(x, y, ci, n) = 0.0f;
    f_output(x, y, ci, n) += 
        f_radiance(x + r.x, y + r.y, ci, n)
      * f_kernel(x, y, r.x, r.y, n);

    std::map<std::string, Func> func_map;
    func_map["radiance"] = f_radiance;
    func_map["kernel"] = f_kernel;
    func_map["output"]  = f_output;

    return func_map;
}
