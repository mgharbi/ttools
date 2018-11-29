#include "gradient_helpers.h"

namespace gradient_apps {

class TempGenerator : public Generator<TempGenerator> {
public:
    Input<Buffer<float>> input{"input", 1};
    Output<Buffer<float>> output{"output", 1};

    void generate() {
      Var x("x");
      RDom r(0, 5);
      output(x) = 0.0f;
      output(r) += input(r);
    }
        
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::TempGenerator, temp)
