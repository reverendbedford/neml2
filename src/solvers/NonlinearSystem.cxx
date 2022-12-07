#include "solvers/NonlinearSystem.h"

BatchTensor<1>
NonlinearSystem::residual(BatchTensor<1> x) const
{
  BatchTensor<1> r = x.clone();
  set_residual(x, r);
  return r;
}

BatchTensor<1>
NonlinearSystem::Jacobian(BatchTensor<1> x) const
{
  auto [r, J] = residual_and_Jacobian(x);
  return J;
}

std::tuple<BatchTensor<1>, BatchTensor<1>>
NonlinearSystem::residual_and_Jacobian(BatchTensor<1> x) const
{
  TorchSize n = x.base_sizes()[0];
  BatchTensor<1> r = x.clone();
  BatchTensor<1> J(x.batch_sizes(), {n, n});
  set_residual(x, r, &J);
  return {r, J};
}
