#pragma once

#include <map>
#include <string>
#include <Halide.h>

#include "helpers.h"

using namespace Halide;

Var x("x"), y("y"), 
    ci("ci"), c("c"), n("n"), k("k"), xk("xk"), yk("yk");

template <typename InputBuffer, typename IntInputBuffer, typename OutputBuffer>
std::map<std::string, Func> kernel_lookup(
        const InputBuffer &data,
        const IntInputBuffer &kernel_idx,
        const InputBuffer &weights,
        const OutputBuffer &output
        )
{
    Func f_data("f_data");
    f_data = Halide::BoundaryConditions::constant_exterior(data, 0.0f);
    Func f_kidx("f_kidx");
    f_kidx = Halide::BoundaryConditions::constant_exterior(kernel_idx, 0);
    Func f_weights("f_weights");
    f_weights = Halide::BoundaryConditions::constant_exterior(weights, 0.0f);

    Expr kw = weights.dim(0).extent();
    Expr kh = weights.dim(1).extent();
    Expr channels = data.dim(2).extent();

    // Reduction over the kernel's extent.
    RDom r_kernel(-(kw-1)/2, kw, -(kh-1)/2, kh, 0, channels);
    Expr dx = r_kernel.x;
    Expr dy = r_kernel.y;
    Expr dc = r_kernel.z;

    // Kernel index to read.
    Expr k_idx = clamp(f_kidx(x, y, c, n), weights.dim(3).min(), weights.dim(3).max());

    // Kernel weight corresponding to the reduction multi-index.
    Expr w = f_weights(dx + (kw-1)/2, dy + (kh-1)/2, dc, k_idx);
    
    // Sum input values in the neighborhood of (x, y) using the appropriate kernel
    output(x, y, c, n) = 0.0f;
    output(x, y, c, n) += w * f_data(x + dx, y + dy, dc, n);

    std::map<std::string, Func> func_map;
    func_map["data_bc"] = f_data;

    return func_map;
}


template <typename InputBuffer, typename IntInputBuffer, typename OutputBuffer>
std::map<std::string, Func> kernel_lookup_backward(
        const InputBuffer &data,
        const IntInputBuffer &kernel_idx,
        const InputBuffer &weights,
        const InputBuffer &d_output,
        const OutputBuffer &d_data,
        const OutputBuffer &d_weights
        )
{
    Func f_data("f_data");
    f_data = Halide::BoundaryConditions::constant_exterior(data, 0.0f);
    Func f_kidx("f_kidx");
    f_kidx = Halide::BoundaryConditions::constant_exterior(kernel_idx, 0);
    Func f_weights("f_weights");
    f_weights = Halide::BoundaryConditions::constant_exterior(weights, 0.0f);
    Func f_d_output("f_d_output");
    f_d_output = Halide::BoundaryConditions::constant_exterior(d_output, 0.0f);

    Expr kw = weights.dim(0).extent();
    Expr kh = weights.dim(1).extent();
    Expr channels = data.dim(2).extent();

    // Reduction over the kernel's extent.
    RDom r_kernel(-(kw-1)/2, kw, -(kh-1)/2, kh, 0, channels);
    Expr dx = r_kernel.x;
    Expr dy = r_kernel.y;
    Expr dc = r_kernel.z;

    // Kernel index to read.
    Expr k_idx = clamp(f_kidx(x-dx, y-dy, dc, n), weights.dim(3).min(), weights.dim(3).max());

    // Kernel weight corresponding to the reduction multi-index.
    Expr w = f_weights(dx + (kw-1)/2, dy + (kh-1)/2, c, k_idx);
    
    // Sum gradients in the neighborhood of (x, y) using the appropriate kernel
    d_data(x, y, c, n) = 0.0f;
    d_data(x, y, c, n) += w * f_d_output(x - dx, y - dy, dc, n);

    // Reduction over the entire image to backprop to weights
    Expr width = d_output.dim(0).extent();
    Expr height = d_output.dim(1).extent();
    Expr o_channels = d_output.dim(2).extent();
    Expr batch = d_output.dim(3).extent();
    RDom r_dkernel(0, width, 0, height, 0, o_channels, 0, batch);
    Expr rx = r_dkernel.x;
    Expr ry = r_dkernel.y;
    Expr rc = r_dkernel.z;
    Expr rn = r_dkernel.w;

    Expr k_idx2 = clamp(f_kidx(rx, ry, rc, rn), weights.dim(3).min(), weights.dim(3).max());
    Expr kernel_matches = select(k_idx2 == k, 1.0f, 0.0f);

    d_weights(xk, yk, c, k) = 0.0f;
    d_weights(xk, yk, c, k) += 
        f_d_output(rx, ry, rc, rn)*f_data(rx + xk, ry + yk, c, rn)*kernel_matches;

    // d_data(x, y, c, n) += w * f_d_output(x - dx, y - dy, dc, n);

    std::map<std::string, Func> func_map;
    return func_map;
}
