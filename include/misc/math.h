#pragma once

#include "misc/types.h"

namespace neml2
{
namespace math
{
/// A batched version of torch::linspace
template <TorchSize N>
BatchTensor<N>
linspace(BatchTensor<N> start, BatchTensor<N> stop, TorchSize nsteps)
{
  neml_assert(start.sizes() == stop.sizes(),
              "start and stop tensors need the same shape in "
              "linspace. The start tensor has shape ",
              start.sizes(),
              " while the stop tensor has shape ",
              stop.sizes());

  auto steps = torch::arange(nsteps) / (nsteps - 1);

  for (int i = 0; i < start.dim(); i++)
    steps = steps.unsqueeze(-1);

  return start.index({torch::indexing::None}) +
         steps * (stop - start).index({torch::indexing::None});
}
} // namespace math
} // namespace neml2
