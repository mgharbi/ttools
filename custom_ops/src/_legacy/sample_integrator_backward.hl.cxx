#include "algorithms/sample_integrator.h"

#include "gradient_helpers.h"

namespace gradient_apps {

class SampleIntegratorBackwardGenerator : public Generator<SampleIntegratorBackwardGenerator> {
public:
    Input<Buffer<float>> samples{"samples", 5};
    Input<Buffer<float>> coordinates{"coordinates", 5};
    Input<Buffer<float>> projections{"projections", 2};
    Input<Buffer<float>> biases{"biases", 1};
    Input<int> nsize{"nsize"};
    Input<Buffer<float>> d_output{"d_output", 4};

    Output<Buffer<float>> d_samples{"d_samples", 5};
    Output<Buffer<float>> d_projections{"d_projections", 2};
    Output<Buffer<float>> d_biases{"d_biases", 1};

    void generate() {
        Expr hsize = (nsize-1) / 2;
        Expr f_hsize = cast<float>(nsize)*0.5f;

        std::map<std::string, Func> funcs = sample_integrator(
            samples, coordinates, projections, biases, nsize);

        Expr spp = samples.dim(0).extent();
        Expr channels_in = samples.dim(3).extent();
        Expr nfilters = biases.dim(0).extent();
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
        Func d_projected_coord("d_projected_coord");
        d_projected_coord(s, dx, dy, x, y, k, n) = 
          d_weights(s, dx, dy, x, y, k, n)*d_sigmoid(
              projected_coord(s, dx, dy, x, y, k, n));

        RDom r(-hsize, nsize, -hsize, nsize, 0, nfilters, "r");  // Sum over samples

        Func weights = funcs["weights"];

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
          { d_projected_coord(s, dx, dy, x, y, k, n),
            d_projected_coord(s, dx, dy, x, y, k, n)*centered_coord(
                s, dx, dy, x, y, coord, n)};

        Func tup_sample_reduced("tup_sample_reduced");
        RDom r1(0, spp, -hsize, nsize, -hsize, nsize);
        tup_sample_reduced(x, y, coord, k, n) = {0.0f, 0.0f};
        tup_sample_reduced(x, y, coord, k, n) += tup_coords(
            r1.x, r1.y, r1.z, x, y, k, coord, n);

        RDom r2(0, samples.dim(1).extent(), 
                0, samples.dim(2).extent(), 
                0, samples.dim(4).extent(), "r2");
        
        Func tup_image_reduce("tup_image_reduce");
        tup_image_reduce(coord, k) = {0.0f, 0.0f};
        tup_image_reduce(coord, k) += 
          tup_sample_reduced(r2.x, r2.y, coord, k, r2.z);

        d_projections(coord, k) = tup_image_reduce(coord, k)[1];
        
        d_biases(k) = tup_image_reduce(0, k)[0];

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

            Var gpu_tile("gpu_tile");
            Var gpu_tile_z("gpu_tile_z");
            Var gpu_tile_y("gpu_tile_y");
            Var gpu_tile_x("gpu_tile_x");
            Var gpu_threads("gpu_threads");
            Var gpu_threads_z("gpu_threads_z");
            Var gpu_threads_y("gpu_threads_y");
            Var gpu_threads_x("gpu_threads_x");

            d_samples
              .gpu_tile(x, y, gpu_tile_x, gpu_tile_y,
                        gpu_threads_x, gpu_threads_y, 4, 4)
              .update()
              .fuse(x, s, x)
              .gpu_tile(x, y, gpu_tile_x, gpu_tile_y, gpu_threads_x,
                        gpu_threads_y, 16, 16)
              ;
            // d_samples.print_loop_nest();

            Var xy("xy");
            Var xyn("xyn");

            projected_coord
              .fuse(n, k, k)
              .fuse(k, y, y)
              .fuse(y, x, x)
              .fuse(x, dy, dy)
              .fuse(dy, dx, dx)
              .fuse(dx, s, s)
              .update()
              .fuse(n, k, k)
              .fuse(k, y, y)
              .fuse(y, x, x)
              .fuse(x, dy, dy)
              .fuse(dy, dx, dx)
              .fuse(dx, s, s)
              ;

            // Fuse the reduction variables on the image + batch
            tup_image_reduce
              .compute_root()
              .update()
              .fuse(r2.x, r2.y, r2.x)
              .fuse(r2.x, r2.z, r2.x)
              .split(r2.x, r2.x, r2.y, 128)
              ;

            // Reduce batches in parallel
            Var r_batches("r_batches");
            Func tup_itm = tup_image_reduce.update().rfactor(r2.y, r_batches);

            // Reduce image in parallel
            Var r_image("r_image");
            Func tup_itm2 = tup_itm.update().rfactor(r2.x, r_image);

            tup_sample_reduced
              .compute_at(tup_image_reduce, coord)
              .fuse(k, coord, k)
              .fuse(x, y, xy)
              .fuse(xy, n, xyn)
              .fuse(xyn, k, xyn)
              .gpu_tile(xyn, xi, 256)
              .update()
              .fuse(k, coord, k)
              .fuse(x, y, xy)
              .fuse(xy, n, xyn)
              .fuse(xyn, k, xyn)
              .gpu_tile(xyn, xi, 256)
              .fuse(r1.x, r1.y, r1.y)
              .fuse(r1.y, r1.z, r1.z)
              ;

            // tup_sample_reduced.print_loop_nest();

            tup_itm2
              .compute_at(tup_image_reduce, coord)
              .fuse(k, coord, k)
              .fuse(k, r_batches, r_batches)
              .fuse(r_batches, r_image, r_image)
              .gpu_tile(r_image, xi, ts)
              .update()
              .fuse(k, coord, k)
              .fuse(k, r_batches, r_batches)
              .fuse(r_batches, r_image, r_image)
              .gpu_tile(r_image, xi, ts)
              ;
            tup_itm
              .compute_at(tup_image_reduce, coord)
              .fuse(k, coord, k)
              .fuse(k, r_batches, r_batches)
              .gpu_tile(r_batches, xi, ts)
              .update()
              .fuse(k, coord, k)
              .fuse(k, r_batches, r_batches)
              .gpu_tile(r_batches, xi, ts)
              ;
            // tup_itm2
            //   .compute_at(tup_itm, r_batches)
            //   .gpu_threads(r_batches)
            //   .update()
            //   .gpu_threads(r_batches)
            //   ;

            // tup_image_reduce.print_loop_nest();


          } else {
            cerr << "cpu schedule\n";
          }
        }
    }
        
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::SampleIntegratorBackwardGenerator, sample_integrator_backward)
