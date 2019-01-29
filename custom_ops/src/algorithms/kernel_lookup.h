#pragma once

#include <map>
#include <string>
#include <Halide.h>

#include "helpers.h"

using namespace Halide;

Var x("x"), y("y"), 
    dx("dx"), dy("dy"),
    ci("ci"), c("c"), n("n");

template <typename InputBuffer, typename IntInputBuffer, typename OutputBuffer>
std::map<std::string, Func> kernel_lookup(
        const InputBuffer &data,
        const IntInputBuffer &kernel_idx,
        const InputBuffer &weights,
        const OutputBuffer &output
        )
{
    Func f_data("f_data");
    f_data(x, y, ci, n) = Halide::BoundaryConditions::constant_exterior(
        data, 0.0f)(x, y, ci, n);
    Func f_weights("f_weights");
    f_weights(dx, dy, c, n) = Halide::BoundaryConditions::constant_exterior(
        weights, 0.0f)(dx, dy, c, n);

    Expr kw = weights.dim(0).extent();
    Expr kh = weights.dim(1).extent();
    Expr channels = data.dim(2).extent();

    // Reduction over the kernel's extent.
    RDom r_kernel(0, kw, 0, kh, 0, channels);

    // Kernel index to read.
    Expr k_idx = kernel_idx(x, y, c, n);

    // Kernel weight corresponding to the reduction multi-index.
    Expr w = f_weights(r_kernel.x, r_kernel.y, r_kernel.z, k_idx);
    
    // Sum input values in the neighborhood of (x, y) using the appropriate kernel
    output(x, y, c, n) = 0.0f;
    output(x, y, c, n) +=
        w * f_data(x + r_kernel.x - (kw-1)/2,
                   y + r_kernel.y - (kh-1)/2,
                   r_kernel.z, n);

    std::map<std::string, Func> func_map;

    return func_map;
}


template <typename InputBuffer, typename IntInputBuffer, typename OutputBuffer>
std::map<std::string, Func> kernel_lookup_backward(
        const InputBuffer &data,
        const IntInputBuffer &kernel_idx,
        const InputBuffer &weights,
        const OutputBuffer &output
        )
{
    Func f_data("f_data");
    f_data(x, y, ci, n) = Halide::BoundaryConditions::constant_exterior(
        data, 0.0f)(x, y, ci, n);
    Func f_weights("f_weights");
    f_weights(dx, dy, c, n) = Halide::BoundaryConditions::constant_exterior(
        weights, 0.0f)(dx, dy, c, n);

    Expr kw = weights.dim(0).extent();
    Expr kh = weights.dim(1).extent();
    Expr channels = data.dim(2).extent();

    // Reduction over the kernel's extent.
    RDom r_kernel(0, kw, 0, kh, 0, channels);

    // Kernel index to read.
    Expr k_idx = kernel_idx(x, y, c, n);

    // Kernel weight corresponding to the reduction multi-index.
    Expr w = f_weights(r_kernel.x, r_kernel.y, r_kernel.z, k_idx);
    
    // Sum input values in the neighborhood of (x, y) using the appropriate kernel
    output(x, y, c, n) = 0.0f;
    output(x, y, c, n) +=
        w * f_data(x + r_kernel.x - (kw-1)/2,
                   y + r_kernel.y - (kh-1)/2,
                   r_kernel.z, n);

    std::map<std::string, Func> func_map;

    return func_map;
}
