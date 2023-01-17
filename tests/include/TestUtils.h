// Copyright 2023, UChicago Argonne, LLC
// All Rights Reserved
// Software Name: NEML2 -- the New Engineering material Model Library, version 2
// By: Argonne National Laboratory
// OPEN SOURCE LICENSE (MIT)
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy 
// of this software and associated documentation files (the "Software"), to deal 
// in the Software without restriction, including without limitation the rights 
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
// copies of the Software, and to permit persons to whom the Software is 
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in 
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
// THE SOFTWARE.

#pragma once

#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/LabeledVector.h"

template <typename F, neml2::TorchSize D>
void
finite_differencing_derivative(F && f,
                               const neml2::LabeledVector & x,
                               neml2::LabeledTensor<1, D> & dy_dx,
                               neml2::Real eps = 1e-6,
                               neml2::Real aeps = 1e-6)
{
  using namespace torch::indexing;

  auto y0 = f(x);

  for (neml2::TorchSize i = 0; i < x.tensor().base_sizes()[0]; i++)
  {
    neml2::Scalar dx = eps * torch::abs(x.tensor().base_index({i})).unsqueeze(-1);
    dx.index_put_({dx < aeps}, aeps);

    auto x1 = x.clone();
    neml2::Scalar x1i = x1.tensor().base_index({i}).unsqueeze(-1);
    x1i.index_put_({None}, x1i + dx);

    auto y1 = f(x1);
    dy_dx.tensor().index_put_({Ellipsis, i}, (y1.tensor() - y0.tensor()) / dx);
  }
}

template <typename F>
void
finite_differencing_derivative(F && f,
                               const neml2::BatchTensor<1> & x,
                               neml2::BatchTensor<1> & dy_dx,
                               neml2::Real eps = 1e-6,
                               neml2::Real aeps = 1e-6)
{
  using namespace torch::indexing;

  neml2::BatchTensor<1> y0 = f(x);

  for (neml2::TorchSize i = 0; i < x.base_sizes()[0]; i++)
  {
    neml2::Scalar dx = eps * torch::abs(x.base_index({i})).unsqueeze(-1);
    dx.index_put_({dx < aeps}, aeps);

    neml2::BatchTensor<1> x1 = x.clone();
    neml2::Scalar x1i = x1.base_index({i}).unsqueeze(-1);
    x1i.index_put_({None}, x1i + dx);

    neml2::BatchTensor<1> y1 = f(x1);
    dy_dx.index_put_({Ellipsis, i}, (y1 - y0) / dx);
  }
}
