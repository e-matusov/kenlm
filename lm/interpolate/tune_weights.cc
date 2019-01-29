#include "lm/interpolate/tune_weights.hh"

#include "lm/interpolate/tune_derivatives.hh"
#include "lm/interpolate/tune_instances.hh"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas" // Older gcc doesn't have "-Wunused-local-typedefs" and complains.
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#include <Eigen/Dense>
#pragma GCC diagnostic pop
#include <boost/program_options.hpp>

#include <iostream>

namespace lm { namespace interpolate {
void TuneWeights(int tune_file, const std::vector<StringPiece> &model_names, const InstancesConfig &config, float step_size, std::vector<float> &weights_out) {
  Instances instances(tune_file, model_names, config);
  Vector weights = Vector::Constant(model_names.size(), 1.0 / model_names.size());
  Vector* best_weights=0;
  Vector gradient;
  Vector perps(10, 1000000);
  Matrix hessian;
  for (std::size_t iteration = 0; iteration < 10; ++iteration) {
    std::cerr << "Iteration " << iteration << ": weights =";
    for (Vector::Index i = 0; i < weights.rows(); ++i) {
      std::cerr << ' ' << weights(i);
    }
    std::cerr << std::endl;
    
    float perp = Derivatives(instances, weights, gradient, hessian);
    std::cerr << "Perplexity = " << perp << std::endl;
    perps[iteration] = perp;
    // TODO: 1.0 step size was too big and it kept getting unstable.  More math.
    weights -= step_size * hessian.inverse() * gradient;
    if((iteration == 0) || (perp < perps[iteration - 1]))
        best_weights = &weights;        
    // STOPPING CRITERION
    if((iteration > 3) && (perp < perps[0]) && (perps[iteration - 1] < perps[iteration - 2]) && (std::abs(perp - perps[iteration - 1]) < 1.0))
      break;      
}
  weights_out.assign(best_weights->data(), best_weights->data() + best_weights->size());
}
}} // namespaces
