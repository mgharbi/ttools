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

    RDom r_kernel(0, kw, 0, kh, 0, channels);
    Expr w = f_weights(r_kernel.x, r_kernel.y, r_kernel.z, n);
    
    output(x, y, c, n) = 0.0f;
    output(x, y, c, n) +=
        w * f_data(x + r_kernel.x - (kw-1)/2, y + r_kernel.y - (kh-1)/2,
                   r_kernel.z, n);

    // Func homogeneous("homegeneous");
    // homogeneous(x, y, c, n) = select(c < channels, f_data(x, y, c, n), 1.0f);

    // Func summed("summed");
    // summed(x, y, c, n) = 0.0f;
    // summed(x, y, c, n) += w
    //     * homogeneous(x + r_kernel.x - (kw-1)/2,
    //                   y + r_kernel.y - (kh-1)/2,
    //                   c, n);

    // output(x, y, c, n) = summed(x, y, c, n);

    // sum_w(x, y, n) = summed(x, y, 3, n);
    

    std::map<std::string, Func> func_map;
    // func_map["summed"] = summed;

    return func_map;
}

// template <typename InputBuffer, typename OutputBuffer>
// std::map<std::string, Func> kernel_weighting_backward(
//         const InputBuffer &data,
//         const InputBuffer &weights,
//         const InputBuffer &sum_w,
//         const InputBuffer &d_output,
//         const InputBuffer &d_sum_w,
//
//         const OutputBuffer &d_data,
//         const OutputBuffer &d_weights
//         )
// {
//     Func f_data("f_data");
//     f_data(x, y, ci, n) = Halide::BoundaryConditions::constant_exterior(
//         data, 0.0f)(x, y, ci, n);
//     Func f_d_output("f_d_output");
//     f_d_output(x, y, c, n) = Halide::BoundaryConditions::constant_exterior(
//         d_output, 0.0f)(x, y, c, n);
//     Func f_weights("f_weights");
//     f_weights(x, y, dx, dy, n) = Halide::BoundaryConditions::constant_exterior(
//         weights, 0.0f)(x, y, dx, dy, n);
//
//     Expr kw = weights.dim(2).extent();
//     Expr kh = weights.dim(3).extent();
//     Expr channels = data.dim(2).extent();
//
//     RDom r_kernel(0, kw, 0, kh);
//    
//     Expr w = f_weights(x + r_kernel.x - (kw-1)/2, 
//                        y + r_kernel.y - (kh-1)/2,
//                        kw - 1 - r_kernel.x,
//                        kh - 1 - r_kernel.y, n);
//
//     // out = sum { data * w }
//     // dL / ddata = sum {dL/dout * dout / ddata } (= sum {dL/dout * w})
//     //              + sum {dL/dsumw * dsumw / ddata} (=0)
//     Func d_data_tmp("d_data_tmp");
//     d_data_tmp(x, y, c, n) = 0.0f;
//     d_data_tmp(x, y, c, n) += w * f_d_output(
//             x + r_kernel.x - (kw-1)/2, y + r_kernel.y - (kh-1)/2, c, n);
//     d_data(x, y, c, n) = d_data_tmp(x, y, c, n);
//
//     // sumw = sum { w }
//     // dL / dwj = sum { dL/dout * dout / dwj } (=sum{dL/dout * dataj})
//     //          + sum { dL/dsumw * dsumw / dwj } (=sum{dL/dsumw * wj})
//     // Expr w2 = f_weights(x, y, dx, dy, n);
//     Func d_weights_tmp("d_weights_tmp");
//     RDom rchan(0, data.dim(2).extent());
//     d_weights_tmp(x, y, dx, dy, n) = d_sum_w(x, y, n);
//     d_weights_tmp(x, y, dx, dy, n) += 
//         f_data( x + dx - (kw-1)/2, y + dy - (kh-1)/2, rchan, n)
//         * f_d_output(x, y, rchan, n);
//     d_weights(x, y, dx, dy, n) = d_weights_tmp(x, y, dx, dy, n);
//     
//     std::map<std::string, Func> func_map;
//     // func_map["d_data_tmp"] = d_data_tmp;
//     // func_map["d_weights_tmp"] = d_weights_tmp;
//
//     return func_map;
// }
