#include "algorithms/kernel_lookup.h"

#include "gradient_helpers.h"

using namespace Halide;

namespace gradient_apps {

class KernelLookupBackwardGenerator : public Generator<KernelLookupBackwardGenerator> {
public:
    Input<Buffer<float>> data{"data", 4};
    Input<Buffer<int>> kernel_idx{"kernel_idx", 4};
    Input<Buffer<float>> weights{"weights", 4};
    Input<Buffer<float>> d_output{"d_output", 4};
    Output<Buffer<float>> d_data{"d_data", 4};
    Output<Buffer<float>> d_weights{"d_weights", 4};

    void generate() {
        Func output("output");
        std::map<std::string, Func> funcs = kernel_lookup_backward(
            data, kernel_idx, weights, d_output, d_data, d_weights);

        Var tx("tx"), ty("ty"), tz("tz"), dxdy("dxdy"),
            xy("xy"), cn("cn"), allvars("allvars");
        int ts = 16;

        Expr kw = weights.dim(0).extent();
        Expr kh = weights.dim(1).extent();
        Expr channels = weights.dim(2).extent();
        Expr nkernels = weights.dim(3).extent();
        Expr small_elt_count = kw*kh*channels*nkernels < 16;

        if(get_target().has_gpu_feature()) {
            d_data
                .fuse(x, y, xy)
                .fuse(c, n, cn)
                .fuse(xy, cn, allvars)
                .gpu_tile(allvars, tx, 1024)
                ;
            d_data
                .update()
                .fuse(x, y, xy)
                .fuse(c, n, cn)
                .fuse(xy, cn, allvars)
                .gpu_tile(allvars, tx, 1024)
                ;
            d_weights
                .fuse(xk, yk, xy)
                .fuse(c, k, cn)
                ;
            d_weights
                .update()
                .fuse(xk, yk, xy)
                .fuse(c, k, cn)
                ;
            d_weights
                .gpu_tile(cn, tx, 4)
                ;
            d_weights
                .update()
                .gpu_tile(cn, tx, 4)
                ;
        } else {
            d_data
                .fuse(c, n, cn)
                .fuse(y, cn, allvars)
                .parallel(allvars, 8)
                .vectorize(x, 8)
                ;
            d_data
                .update()
                .fuse(c, n, cn)
                .fuse(y, cn, allvars)
                .parallel(allvars, 8)
                .vectorize(x, 8)
                ;
        }
        // d_data.print_loop_nest();
        // d_weights.print_loop_nest();
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::KernelLookupBackwardGenerator, kernel_lookup_backward)
