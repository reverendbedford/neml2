#include "SampleNonlinearSystems.h"

using namespace torch::indexing;

PowerTestSystem::PowerTestSystem() {}

void
PowerTestSystem::set_residual(BatchTensor<1> x,
                              BatchTensor<1> residual,
                              BatchTensor<1> * Jacobian) const
{
  TorchSize n = x.base_sizes()[0];
  for (TorchSize i = 0; i < n; i++)
    residual.base_index_put({i}, x.base_index({i}).pow(i + 1) - 1.0);

  if (Jacobian)
    for (TorchSize i = 0; i < n; i++)
      Jacobian->base_index_put({i, i}, (i + 1) * x.base_index({i}).pow(i));
}

BatchTensor<1>
PowerTestSystem::exact_solution(BatchTensor<1> x) const
{
  return torch::ones_like(x, TorchDefaults);
}

BatchTensor<1>
PowerTestSystem::guess(BatchTensor<1> x) const
{
  return torch::ones_like(x, TorchDefaults) * 2;
}
