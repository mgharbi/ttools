#include "algorithms/kernel_weighting.h"

#include "gradient_helpers.h"

namespace rendernet {

class KernelWeightingForwardGenerator : public Generator<KernelWeightingForwardGenerator> {
public:
    Input<Buffer<float>> data{"data", 4};
    Input<Buffer<float>> weights{"weights", 5};
    Output<Buffer<float>> output{"output", 4};
    Output<Buffer<float>> sum_w{"sum_w", 3};

    void generate() {
        std::map<std::string, Func> funcs = kernel_weighting(
            data, weights, output, sum_w);

        Var tx("tx"), ty("ty"), tz("tz"),
            xy("xy"), cn("cn"), allvars("allvars");
        int ts = 16;


        if(get_target().has_gpu_feature()) {
            output
                .fuse(x, y, xy)
                .fuse(c, n, cn)
                .fuse(xy, cn, allvars)
                .gpu_tile(allvars, tx, 1024)
                ;

            sum_w
                .fuse(x, y, xy)
                .fuse(xy, n, allvars)
                .gpu_tile(allvars, tx, 1024)
                ;

            funcs["summed"]
                .compute_root()
                .gpu_tile(x, y, tx, ty, ts, ts)
                .update()
                .gpu_tile(x, y, tx, ty, ts, ts)
                ;
        } else {
            output
                .compute_root()
                .fuse(c, n, cn)
                .fuse(y, cn, allvars)
                .parallel(allvars, 8)
                .vectorize(x, 8)
                ;

            sum_w
                .compute_root()
                .fuse(y, n, allvars)
                .parallel(allvars, 8)
                .vectorize(x, 8)
                ;

            funcs["summed"]
                .compute_root()
                .parallel(y, 8)
                .vectorize(x, 8)
                .update()
                .parallel(y, 8)
                .vectorize(x, 8)
                ;
        }
    }

};

}  // end namespace rendernet

HALIDE_REGISTER_GENERATOR(
        rendernet::KernelWeightingForwardGenerator, kernel_weighting_forward)
