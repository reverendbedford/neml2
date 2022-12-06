#pragma once

#include "tensors/Scalar.h"
#include "tensors/LabeledVector.h"

/// Derivative of a LabeledVector(LabeledVector) function
template <typename F, TorchSize D>
void
finite_differencing_derivative(
    F && f, const LabeledVector & x, LabeledTensor<1, D> & dy_dx, Real eps = 1e-6, Real aeps = 1e-6)
{
  using namespace torch::indexing;

  auto y0 = f(x);

  for (TorchSize i = 0; i < x.tensor().base_sizes()[0]; i++)
  {
    Scalar dx = eps * torch::abs(x.tensor().base_index({i})).unsqueeze(-1);
    dx.index_put_({dx < aeps}, aeps);

    auto x1 = x.clone();
    Scalar x1i = x1.tensor().base_index({i}).unsqueeze(-1);
    x1i.index_put_({None}, x1i + dx);

    auto y1 = f(x1);
    dy_dx.tensor().index_put_({Ellipsis, i}, (y1.tensor() - y0.tensor()) / dx);
  }
}

/// Derivative of a BatchTensor<1>(BatchTensor<1>) function
template <typename F>
void
finite_differencing_derivative(
    F && f, const BatchTensor<1> & x, BatchTensor<1> & dy_dx, Real eps = 1e-6, Real aeps = 1e-6)
{
  using namespace torch::indexing;

  BatchTensor<1> y0 = f(x);

  for (TorchSize i = 0; i < x.base_sizes()[0]; i++)
  {
    Scalar dx = eps * torch::abs(x.base_index({i})).unsqueeze(-1);
    dx.index_put_({dx < aeps}, aeps);

    BatchTensor<1> x1 = x.clone();
    Scalar x1i = x1.base_index({i}).unsqueeze(-1);
    x1i.index_put_({None}, x1i + dx);

    BatchTensor<1> y1 = f(x1);
    dy_dx.index_put_({Ellipsis, i}, (y1 - y0) / dx);
  }
}
