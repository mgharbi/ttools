#include "algorithms/kernel_reduction.h"

#include "gradient_helpers.h"

namespace gradient_apps {

class KernelReductionForwardGenerator : public Generator<KernelReductionForwardGenerator> {
public:
    Input<Buffer<float>> radiance{"radiance", 4};
    Input<Buffer<float>> kernel{"kernel", 5};
    Output<Buffer<float>> output{"output", 4};

    void generate() {
        std::map<std::string, Func> funcs = kernel_reduction(
            radiance, kernel);

        output(x, y, ci, n) = funcs["output"](x, y, ci, n);

        if (get_target().has_gpu_feature()) {
          Var xi("xi"), yi("yi");
          output.compute_root().gpu_tile(x, y, xi, yi, 8, 8);
          // funcs["output"].
        }

        // SimpleAutoscheduleOptions options;
        // Func output_func = output;
        //
        // std::set<std::string> dont_inline = {};
        //
        // simple_autoschedule(output_func,
        //     {
        //     {"radiance.min.0", 0},
        //     {"radiance.min.1", 0},
        //     {"radiance.min.2", 0},
        //     {"radiance.min.3", 0},
        //     {"radiance.extent.0", 128},
        //     {"radiance.extent.1", 128},
        //     {"radiance.extent.2", 3},
        //     {"radiance.extent.3", 4},
        //     {"kernel.min.0", 0},
        //     {"kernel.min.1", 0},
        //     {"kernel.min.2", 0},
        //     {"kernel.min.3", 0},
        //     {"kernel.min.4", 0},
        //     {"kernel.extent.0", 128},
        //     {"kernel.extent.1", 128},
        //     {"kernel.extent.2", 21},
        //     {"kernel.extent.3", 21},
        //     {"kernel.extent.4", 4}
        //     },
        //     {{0, 127},
        //       {0, 127},
        //       {0, 2},
        //       {0, 3}},
        //     options,
        //     dont_inline);
    }
        
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::KernelReductionForwardGenerator, kernel_reduction_forward)
