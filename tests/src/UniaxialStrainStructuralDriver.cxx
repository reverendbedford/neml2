#include "UniaxialStrainStructuralDriver.h"

using namespace neml2;

using namespace torch::indexing;

UniaxialStrainStructuralDriver::UniaxialStrainStructuralDriver(const Model & model,
                                                               Scalar max_strain,
                                                               Scalar end_time,
                                                               TorchSize nsteps)
  : StructuralStrainControlDriver(
        model,
        batched_linspace(torch::zeros_like(end_time), end_time, nsteps),
        batched_linspace(torch::zeros({max_strain.sizes()[0], 6}),
                         torch::hstack({max_strain,
                                        -0.5 * max_strain,
                                        -0.5 * max_strain,
                                        torch::zeros({max_strain.sizes()[0], 3})}),
                         nsteps)),
    _max_strain(max_strain),
    _end_time(end_time)
{
}

torch::Tensor
batched_linspace(torch::Tensor start, torch::Tensor stop, neml2::TorchSize nsteps)
{
  neml_assert(start.sizes() == stop.sizes(), "start and stop tensors need the same shape");

  auto steps = torch::arange(nsteps) / (nsteps - 1);

  for (int i = 0; i < start.dim(); i++)
    steps = steps.unsqueeze(-1);

  return start.index({None}) + steps * (stop - start).index({None});
}
