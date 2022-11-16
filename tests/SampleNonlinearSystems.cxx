#include "SampleNonlinearSystems.h"

using namespace torch::indexing;

PowerTestSystem::PowerTestSystem(TorchSize nbatch, TorchSize n)
  : _nbatch(nbatch),
    _n(n)
{
}

torch::Tensor
PowerTestSystem::residual(torch::Tensor x)
{
  torch::Tensor result = torch::zeros({_nbatch, _n});
  for (TorchSize i = 0; i < _n; i++)
  {
    result.index_put_({Ellipsis, i}, x.index({Ellipsis, i}).pow(i + 1) - 1.0);
  }

  return result;
}

torch::Tensor
PowerTestSystem::jacobian(torch::Tensor x)
{
  torch::Tensor result = torch::zeros({_nbatch, _n, _n});

  for (TorchSize i = 0; i < _n; i++)
  {
    result.index_put_({Ellipsis, i, i}, (i + 1) * x.index({Ellipsis, i}).pow(i));
  }

  return result;
}

torch::Tensor
PowerTestSystem::exact_solution() const
{
  return torch::ones({_nbatch, _n});
}

torch::Tensor
PowerTestSystem::guess() const
{
  return torch::ones({_nbatch, _n}) * 2.0;
}
