#include "algorithms/sample_integrator.h"

#include "gradient_helpers.h"

namespace gradient_apps {

class SampleIntegratorForwardGenerator : public Generator<SampleIntegratorForwardGenerator> {
public:
    Input<Buffer<float>> samples{"samples", 5};
    Input<Buffer<float>> coordinates{"coordinates", 5};
    Input<Buffer<float>> projections{"projections", 2};
    Input<Buffer<float>> biases{"biases", 1};
    Input<int> nsize{"nsize"};
    Output<Buffer<float>> output{"output", 4};

    void generate() {
        std::map<std::string, Func> funcs = sample_integrator(
            samples, coordinates, projections, biases, nsize);

        output(x, y, co, n) = funcs["output"](x, y, co, n);

        if(auto_schedule) {
        } else {
          Var nc("nc");
          Var ncy("ncy");
          Var ncyx("ncyx");
          Var yi("yi");
          Var xi("xi");

          if (get_target().has_gpu_feature()) {
            cerr << "gpu schedule\n";

            Var gpu_tile("gpu_tile");
            Var gpu_tile_z("gpu_tile_z");
            Var gpu_tile_y("gpu_tile_y");
            Var gpu_tile_x("gpu_tile_x");
            Var gpu_threads("gpu_threads");
            Var gpu_threads_z("gpu_threads_z");
            Var gpu_threads_y("gpu_threads_y");
            Var gpu_threads_x("gpu_threads_x");

            funcs["filtered"]
              .compute_root()
              .gpu_tile(x, y, ci, gpu_tile_x, gpu_tile_y, gpu_tile_z,
                        gpu_threads_x, gpu_threads_y, gpu_threads_z, 4, 4, 2)
              .update()
              .gpu_tile(x, y, ci, gpu_tile_x, gpu_tile_y, gpu_tile_z,
                        gpu_threads_x, gpu_threads_y, gpu_threads_z, 4, 4, 2)
              ;
            output
              .fuse(n, co, nc)
              .fuse(nc, y, ncy)
              .fuse(ncy, x, ncyx)
              .gpu_tile(ncyx, xi, 32)
              ;
            // output.print_loop_nest();
            // print_func(output, true, 3);

          } else {
            cerr << "cpu schedule\n";

            funcs["weights"]
              .compute_at(output, x)
              ;
            output
              .parallel(n)
              // .vectorize(x)
              ;

          }
        }
    }
        
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::SampleIntegratorForwardGenerator, sample_integrator_forward)
