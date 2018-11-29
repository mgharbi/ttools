#pragma once

#include <map>
#include <string>
#include <Halide.h>

#include "helpers.h"

using namespace Halide;

Var s("s"), x("x"), y("y"), 
    dx("dx"), dy("dy"),
    ci("ci"), co("co"), coord("coord"),
    n("n"), k("k");

template <typename InputBuffer>
std::map<std::string, Func> sample_integrator(
        const InputBuffer &samples,
        const InputBuffer &coordinates,
        const InputBuffer &projections,
        const InputBuffer &biases,
        const GeneratorInput<int> &nsize
        ) {
    Func f_samples("f_samples");
    f_samples(s, x, y, ci, n) = Halide::BoundaryConditions::constant_exterior(
        samples, 0.0f)(s, x, y, ci, n);
    Func f_coords("f_coords");
    f_coords(s, x, y, coord, n) = Halide::BoundaryConditions::constant_exterior(
        coordinates, 0.0f)(s, x, y, coord, n);
    Func f_projections("f_projections");
    f_projections(coord, k) = projections(coord, k);
    Func f_biases("f_biases");
    f_biases(k) = biases(k);

    Expr hsize = (nsize-1) / 2;
    Expr f_hsize = cast<float>(nsize)*0.5f;

    // Center local x, y
    Func centered_coord("centered_coord");
    centered_coord(s, dx, dy, x, y, coord, n) =
      select(coord == 0, (f_coords(s, x+dx, y+dy, coord, n) + dx - 0.5f)/f_hsize,
             coord == 1, (f_coords(s, x+dx, y+dy, coord, n) + dy - 0.5f)/f_hsize,
             f_coords(s, x+dx, y+dy, coord, n));
    
    // TODO: maybe add an x,y kernel?

    Expr n_coords = coordinates.dim(3).extent();
    RDom r_coord(0, n_coords);  // project coordinates to 1D subspace

    Func projected_coord("projected_coord");
    projected_coord(s, dx, dy, x, y, k, n) = f_biases(k);
    projected_coord(s, dx, dy, x, y, k, n) += 
      f_projections(r_coord, k)*centered_coord(s, dx, dy, x, y, r_coord, n);

    Func weights("weights");
    weights(s, dx, dy, x, y, k, n) = 
      sigmoid(projected_coord(s, dx, dy, x, y, k, n));  // Weight half-space

    Expr spp = samples.dim(0).extent();
    RDom r(0, spp, -hsize, nsize, -hsize, nsize);  // Sum over samples

    Func filtered("filtered");
    Expr normalizer = 1.0f / (cast<float>(nsize)*cast<float>(nsize)*cast<float>(spp));
    filtered(x, y, ci, k, n) = 0.0f;
    filtered(x, y, ci, k, n) += 
      normalizer
      * weights(r.x, r.y, r.z, x, y, k, n)
      * f_samples(r.x, x+r.y, y+r.z, ci, n);

    Expr channels_in = samples.dim(3).extent();
    Func f_output("f_output");
    f_output(x, y, co, n) = filtered(x, y, co % channels_in, co / channels_in, n);

    std::map<std::string, Func> func_map;
    func_map["samples"] = f_samples;
    func_map["coords"] = f_coords;
    func_map["projections"] = f_projections;
    func_map["biases"] = f_biases;

    func_map["centered_coord"] = centered_coord;
    func_map["projected_coord"] = projected_coord;
    func_map["weights"] = weights;
    func_map["filtered"]  = filtered;
    func_map["output"]  = f_output;

    return func_map;
}
