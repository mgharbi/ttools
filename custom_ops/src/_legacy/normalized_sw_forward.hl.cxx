#include "algorithms/normalized_sw.h"

#include "gradient_helpers.h"

namespace gradient_apps {

class NormalizedSampleWeightingForwardGenerator : public Generator<NormalizedSampleWeightingForwardGenerator> {
public:
    Input<Buffer<float>> samples{"samples", 5};
    Input<Buffer<float>> coordinates{"coordinates", 5};
    Input<Buffer<float>> projections{"projections", 5};
    Input<int> nsize{"nsize"};
    Input<float> sigma2{"sigma2"};
    Output<Buffer<float>> filtered{"filtered", 5};
    Output<Buffer<float>> filtered_weights{"filtered_weights", 5};

    void generate() {
        std::map<std::string, Func> funcs = normalized_sw(
            samples, coordinates, projections, nsize, sigma2);

        filtered(x, y, ci, k, n) = funcs["filtered"](x, y, ci, k, n)[0];
        filtered_weights(x, y, ci, k, n) = funcs["filtered"](x, y, ci, k, n)[1];

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

            funcs["projected_coord"]
              .compute_at(funcs["filtered"], gpu_tile_x)
              .fuse(n, k, gpu_threads)
              .fuse(gpu_threads, x, gpu_threads)
              .fuse(gpu_threads, y, gpu_threads)
              .gpu_threads(gpu_threads)
              .update()
              .fuse(n, k, gpu_threads)
              .fuse(gpu_threads, x, gpu_threads)
              .fuse(gpu_threads, y, gpu_threads)
              .gpu_threads(gpu_threads)
              ;

            std::vector<RVar> r = funcs["filtered"].rvars();
            funcs["filtered"]
              .compute_root()
              .gpu_tile(x, y, ci, gpu_tile_x, gpu_tile_y, gpu_tile_z,
                        gpu_threads_x, gpu_threads_y, gpu_threads_z, 4, 4, 2)
              .update()
              .gpu_tile(x, y, ci, gpu_tile_x, gpu_tile_y, gpu_tile_z,
                        gpu_threads_x, gpu_threads_y, gpu_threads_z, 4, 4, 2)
              ;

            filtered
              .fuse(n, ci, nc)
              .fuse(nc, k, nc)
              .fuse(nc, y, ncy)
              .fuse(ncy, x, ncyx)
              .gpu_tile(ncyx, xi, 32)
              ;

            filtered_weights
              .fuse(n, ci, nc)
              .fuse(nc, k, nc)
              .fuse(nc, y, ncy)
              .fuse(ncy, x, ncyx)
              .gpu_tile(ncyx, xi, 32)
              ;

          } else {
            cerr << "cpu schedule\n";
            filtered
              .parallel(n)
              .vectorize(x, 8)
              ;
            filtered_weights
              .parallel(n)
              .vectorize(x, 8)
              ;

          }
        }
    }
        
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::NormalizedSampleWeightingForwardGenerator, normalized_sw_forward)
