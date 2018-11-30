#include "algorithms/scatter2gather.h"

#include "gradient_helpers.h"

namespace rendernet {

/**
 * Converts sample-centered kernels into pixel-centered kernels.
 */
class Scatter2GatherForwardGenerator : public Generator<Scatter2GatherForwardGenerator> {
public:
    Input<Func> weights{"weights", Float(32), 5};
    Output<Func> output{"output", Float(32), 5};
    Input<int> kw{"kw"};
    Input<int> kh{"kh"};

    void generate() {
        std::map<std::string, Func> funcs = scatter2gather(
            weights, output, kw, kh);

        Var tx("tx"), ty("ty"), tz("tz"), dxdy("dxdy"),
            xy("xy"), cn("cn"), allvars("allvars");
        int ts = 16;

        if(get_target().has_gpu_feature()) {
            output
                .compute_root()
                .fuse(x, y, xy)
                .fuse(dx, dy, dxdy)
                .fuse(xy, dxdy, allvars)
                .fuse(allvars, n, allvars)
                .gpu_tile(allvars, tx, 1024)
                ;
        } else {
            output
                .compute_root()
                .fuse(dx, dy, dxdy)
                .fuse(y, dxdy, allvars)
                .fuse(allvars, n, allvars)
                .parallel(allvars, 8)
                .vectorize(x, 8)
                ;
        }
    }

};

}  // end namespace rendernet

HALIDE_REGISTER_GENERATOR(
        rendernet::Scatter2GatherForwardGenerator, scatter2gather_forward)
