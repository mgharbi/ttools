#include "algorithms/sample_weighting.h"

#include "gradient_helpers.h"

namespace gradient_apps {

class SampleWeightingBackwardGenerator : public Generator<SampleWeightingBackwardGenerator> {
public:
    Input<Buffer<float>> samples{"samples", 5};
    Input<Buffer<float>> coordinates{"coordinates", 5};
    Input<Buffer<float>> projections{"projections", 5};
    Input<int> nsize{"nsize"};
    Input<float> sigma2{"sigma2"};
    Input<Buffer<float>> d_output{"d_output", 4};

    Output<Buffer<float>> d_samples{"d_samples", 5};
    Output<Buffer<float>> d_projections{"d_projections", 5};

    void generate() {
        Expr hsize = (nsize-1) / 2;

        std::map<std::string, Func> funcs = sample_weighting(
            samples, coordinates, projections, nsize, sigma2);

        Expr spp = samples.dim(0).extent();
        Expr channels_in = samples.dim(3).extent();
        // TODO: nfilters probably wrong
        Expr nfilters = projections.dim(1).extent();
        Expr normalizer = 1.0f / (cast<float>(nsize)*cast<float>(nsize)*cast<float>(spp));

        Func f_d_output("f_d_output");
        f_d_output(x, y, co, n) = Halide::BoundaryConditions::constant_exterior(
            d_output, 0.0f)(x, y, co, n);

        Func d_filtered("d_filtered");
        d_filtered(x, y, ci, k, n) = f_d_output(x, y, ci + channels_in*k, n);

        Func f_samples = funcs["samples"];
        Func d_weights("d_weights");
        RDom rci(0, channels_in);
        d_weights(s, dx, dy, x, y, k, n) = 0.0f;
        d_weights(s, dx, dy, x, y, k, n) += 
          normalizer*d_filtered(x, y, rci, k, n)*f_samples(s, x+dx, y+dy, rci, n);

        Func projected_coord = funcs["projected_coord"];

        Expr kernel_x = projected_coord(s, dx, dy, x, y, 2*(k/2), n);
        Expr kernel_y = projected_coord(s, dx, dy, x, y, 2*(k/2) + 1, n);
        Expr weights2 = 
          exp(-0.5f*(kernel_x*kernel_x + kernel_y*kernel_y)/sigma2);

        // derivative of Gaussian
        Func weights = funcs["weights"];
        Func d_projected_coord("d_projected_coord");
        d_projected_coord(s, dx, dy, x, y, k, n) = 
          d_weights(s, dx, dy, x, y, k/2, n) * weights2
          *(-1.0f/sigma2*select(k % 2 == 0, 
                projected_coord(s, dx, dy, x, y, 2*(k/2), n),
                projected_coord(s, dx, dy, x, y, 2*(k/2)+1, n)));  // NOTE: 2*(k/2) matches kernel_x, this is to avoid "dynamic size in cuda block"

        // TODO: nfilters probably wrong
        RDom r(-hsize, nsize, -hsize, nsize, 0, nfilters, "r");  // Sum over samples

        d_samples(s, x, y, ci, n) = 0.0f;
        d_samples(s, x, y, ci, n) += 
          normalizer
          * weights(s, r.x, r.y, x-r.x, y-r.y, r.z, n)
          * d_filtered(x-r.x, y-r.y, ci, r.z, n);
        
        // Center local x, y
        Func centered_coord = funcs["centered_coord"];
        Func f_coords = funcs["coords"];

        Func tup_coords("tup_coords");
        tup_coords(s, dx, dy, x, y, k, coord, n) =
            d_projected_coord(s, dx, dy, x, y, k, n)*centered_coord(
                s, dx, dy, x, y, coord, n);

        RDom r1(0, spp, -hsize, nsize, -hsize, nsize);
        d_projections(x, y, coord, k, n) = 0.0f;
        d_projections(x, y, coord, k, n) += tup_coords(
            r1.x, r1.y, r1.z, x, y, k, coord, n);

        if(auto_schedule) {
        } else {
          Var nc("nc");
          Var ncy("ncy");
          Var ncyx("ncyx");
          Var ncyxs("ncyxs");
          Var yi("yi");
          Var xi("xi");

          if (get_target().has_gpu_feature()) {
            cerr << "gpu schedule\n";
            int ts = 8;

            Var xy("xy");
            Var xyn("xyn");

            Var gpu_tile("gpu_tile");
            Var gpu_tile_z("gpu_tile_z");
            Var gpu_tile_y("gpu_tile_y");
            Var gpu_tile_x("gpu_tile_x");
            Var gpu_threads("gpu_threads");
            Var gpu_threads_z("gpu_threads_z");
            Var gpu_threads_y("gpu_threads_y");
            Var gpu_threads_x("gpu_threads_x");
            Var linear("linear");

            // d_weights
            //   .compute_at(d_projections, gpu_tile_x)
            //   .fuse(n, k, gpu_threads)
            //   .fuse(gpu_threads, x, gpu_threads)
            //   .fuse(gpu_threads, y, gpu_threads)
            //   .gpu_threads(gpu_threads)
            //   .update()
            //   .fuse(n, k, gpu_threads)
            //   .fuse(gpu_threads, x, gpu_threads)
            //   .fuse(gpu_threads, y, gpu_threads)
            //   .gpu_threads(gpu_threads)
            //  ;

             // projected_coord
             //  .clone_in(d_projected_coord)
             //  .compute_at(d_projections, r1.x)
              // .compute_at(d_projections, gpu_tile_x)
              // .fuse(n, k, gpu_threads)
              // .fuse(gpu_threads, x, gpu_threads)
              // .fuse(gpu_threads, y, gpu_threads)
              // .gpu_threads(gpu_threads)
              // .update()
              // .fuse(n, k, gpu_threads)
              // .fuse(gpu_threads, x, gpu_threads)
              // .fuse(gpu_threads, y, gpu_threads)
              // .gpu_threads(gpu_threads)
              // ;

            // TODO: rfactor?
            d_projections
              .compute_root()
              .gpu_tile(x, y, gpu_tile_x, gpu_tile_y,
                        gpu_threads_x, gpu_threads_y, 4, 4)
              .update()
              .fuse(y, coord, y)
              .gpu_tile(x, y, k, gpu_tile_x, gpu_tile_y, gpu_tile_z,
                        gpu_threads_x, gpu_threads_y, gpu_threads_z, 16, 16, 2)
              ;

            d_samples
              .gpu_tile(x, y, gpu_tile_x, gpu_tile_y,
                        gpu_threads_x, gpu_threads_y, 4, 4)
              .update()
              .fuse(x, s, x)
              .gpu_tile(x, y, gpu_tile_x, gpu_tile_y, gpu_threads_x,
                        gpu_threads_y, 16, 16)
              ;

            // d_samples.print_loop_nest();
            // d_projections.print_loop_nest();

          } else {
            cerr << "cpu schedule\n";
          }
        }
    }
        
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::SampleWeightingBackwardGenerator, sample_weighting_backward)
