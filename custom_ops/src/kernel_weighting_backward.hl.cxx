#include "algorithms/kernel_weighting.h"

#include "gradient_helpers.h"

namespace gradient_apps {

class KernelWeightingBackwardGenerator : public Generator<KernelWeightingBackwardGenerator> {
public:
    Input<Buffer<float>> data{"data", 4};
    Input<Buffer<float>> weights{"weights", 5};
    Input<Buffer<float>> sum_w{"sum_w", 3};
    Input<Buffer<float>> d_output{"d_output", 4};
    Input<Buffer<float>> d_sum_w{"d_sum_w", 3};

    Output<Buffer<float>> d_data{"d_data", 4};
    Output<Buffer<float>> d_weights{"d_weights", 5};

    void generate() {
        std::map<std::string, Func> funcs = kernel_weighting_backward(
            data, weights, sum_w, d_output, d_sum_w, d_data, d_weights);

        Var tx("tx"), ty("ty"), tz("tz"), dxdy("dxdy"),
            xy("xy"), cn("cn"), allvars("allvars");
        int ts = 16;

        if(get_target().has_gpu_feature()) {
            d_data
                .gpu_tile(x, y, tx, ty, 32, 32)
                ;
            d_weights
                .gpu_tile(x, y, tx, ty, 32, 32)
                ;
        } else {
            d_data
                .compute_root()
                .fuse(c, n, cn)
                .fuse(y, cn, allvars)
                .parallel(allvars, 8)
                .vectorize(x, 8)
                ;

            d_weights
                .compute_root()
                .fuse(dx, dy, dxdy)
                .fuse(y, dxdy, allvars)
                .fuse(allvars, n, allvars)
                .parallel(allvars, 8)
                .vectorize(x, 8)
                ;
        }
        // d_data.print_loop_nest();

    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::KernelWeightingBackwardGenerator, kernel_weighting_backward)
