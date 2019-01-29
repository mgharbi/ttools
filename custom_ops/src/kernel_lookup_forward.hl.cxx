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
            // output
            //     .fuse(x, y, xy)
            //     .fuse(c, n, cn)
            //     .fuse(xy, cn, allvars)
            //     .gpu_tile(allvars, tx, 1024)
            //     ;
            //
            // sum_w
            //     .fuse(x, y, xy)
            //     .fuse(xy, n, allvars)
            //     .gpu_tile(allvars, tx, 1024)
            //     ;
            //
            // funcs["summed"]
            //     .compute_root()
            //     .gpu_tile(x, y, tx, ty, ts, ts)
            //     .update()
            //     .gpu_tile(x, y, tx, ty, ts, ts)
            //     ;
        } else {
            // output
            //     .compute_root()
            //     .fuse(c, n, cn)
            //     .fuse(y, cn, allvars)
            //     .parallel(allvars, 8)
            //     .vectorize(x, 8)
            //     ;
            //
            // sum_w
            //     .compute_root()
            //     .fuse(y, n, allvars)
            //     .parallel(allvars, 8)
            //     .vectorize(x, 8)
            //     ;
            //
            // funcs["summed"]
            //     .compute_root()
            //     .parallel(y, 8)
            //     .vectorize(x, 8)
            //     .update()
            //     .parallel(y, 8)
            //     .vectorize(x, 8)
            //     ;
        }
    }

};

}  // end namespace rendernet

HALIDE_REGISTER_GENERATOR(
        rendernet::KernelLookupForwardGenerator, kernel_lookup_forward)
