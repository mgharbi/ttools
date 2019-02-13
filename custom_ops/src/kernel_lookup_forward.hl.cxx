#include "algorithms/kernel_lookup.h"

#include "gradient_helpers.h"

namespace rendernet {

class KernelLookupForwardGenerator : public Generator<KernelLookupForwardGenerator> {
public:
    Input<Buffer<float>> data{"data", 4};
    Input<Buffer<int>> kernel_idx{"kernel_idx", 4};
    Input<Buffer<float>> weights{"weights", 4};
    Output<Buffer<float>> output{"output", 4};


    void generate() {
        std::map<std::string, Func> funcs = kernel_lookup(
            data, kernel_idx, weights, output);

        Var tx("tx"), ty("ty"), tz("tz"),
            xy("xy"), cn("cn"), allvars("allvars");
        int ts = 16;

        if(get_target().has_gpu_feature()) {
            output
                .fuse(c, n, cn)
                .gpu_tile(x, y, cn, tx, ty, tz, 32, 32, 1, Halide::TailStrategy::GuardWithIf)
                ;
            output
                .update()
                .fuse(c, n, cn)
                .gpu_tile(x, y, cn, tx, ty, tz, 32, 32, 1, Halide::TailStrategy::GuardWithIf)
                ;
        } else {
            output
                .fuse(c, n, cn)
                .fuse(y, cn, allvars)
                .parallel(allvars, 8)
                .vectorize(x, 8)
                ;
        }
    }

};

}  // end namespace rendernet

HALIDE_REGISTER_GENERATOR(
        rendernet::KernelLookupForwardGenerator, kernel_lookup_forward)
