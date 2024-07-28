// Copyright 2024, UChicago Argonne, LLC
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

#include "neml2/tensors/Tensor.h"

namespace neml2
{
Tensor
Tensor::empty(const TensorShapeRef & base_shape, const torch::TensorOptions & options)
{
  return Tensor(torch::empty(base_shape, options), 0);
}

Tensor
Tensor::empty(const TensorShapeRef & batch_shape,
              const TensorShapeRef & base_shape,
              const torch::TensorOptions & options)
{
  return Tensor(torch::empty(utils::add_shapes(batch_shape, base_shape), options),
                batch_shape.size());
}

Tensor
Tensor::zeros(const TensorShapeRef & base_shape, const torch::TensorOptions & options)
{
  return Tensor(torch::zeros(base_shape, options), 0);
}

Tensor
Tensor::zeros(const TensorShapeRef & batch_shape,
              const TensorShapeRef & base_shape,
              const torch::TensorOptions & options)
{
  return Tensor(torch::zeros(utils::add_shapes(batch_shape, base_shape), options),
                batch_shape.size());
}

Tensor
Tensor::ones(const TensorShapeRef & base_shape, const torch::TensorOptions & options)
{
  return Tensor(torch::ones(base_shape, options), 0);
}

Tensor
Tensor::ones(const TensorShapeRef & batch_shape,
             const TensorShapeRef & base_shape,
             const torch::TensorOptions & options)
{
  return Tensor(torch::ones(utils::add_shapes(batch_shape, base_shape), options),
                batch_shape.size());
}

Tensor
Tensor::full(const TensorShapeRef & base_shape, Real init, const torch::TensorOptions & options)
{
  return Tensor(torch::full(base_shape, init, options), 0);
}

Tensor
Tensor::full(const TensorShapeRef & batch_shape,
             const TensorShapeRef & base_shape,
             Real init,
             const torch::TensorOptions & options)
{
  return Tensor(torch::full(utils::add_shapes(batch_shape, base_shape), init, options),
                batch_shape.size());
}

Tensor
Tensor::identity(Size n, const torch::TensorOptions & options)
{
  return Tensor(torch::eye(n, options), 0);
}

Tensor
Tensor::identity(const TensorShapeRef & batch_shape, Size n, const torch::TensorOptions & options)
{
  return identity(n, options).batch_expand_copy(batch_shape);
}

namespace math
{
Tensor
bmm(const Tensor & a, const Tensor & b)
{
  neml_assert_batch_broadcastable_dbg(a, b);
  neml_assert_dbg(a.base_dim() == 2,
                  "The first tensor in bmm has base dimension ",
                  a.base_dim(),
                  " instead of 2.");
  neml_assert_dbg(b.base_dim() == 2,
                  "The second tensor in bmm has base dimension ",
                  b.base_dim(),
                  " instead of 2.");
  return Tensor(torch::matmul(a, b), broadcast_batch_dim(a, b));
}

Tensor
bmv(const Tensor & a, const Tensor & v)
{
  neml_assert_batch_broadcastable_dbg(a, v);
  neml_assert_dbg(a.base_dim() == 2,
                  "The first tensor in bmv has base dimension ",
                  a.base_dim(),
                  " instead of 2.");
  neml_assert_dbg(v.base_dim() == 1,
                  "The second tensor in bmv has base dimension ",
                  v.base_dim(),
                  " instead of 1.");
  return Tensor(torch::matmul(a, v.base_unsqueeze(-1)).squeeze(-1), broadcast_batch_dim(a, v));
}

Tensor
bvv(const Tensor & a, const Tensor & b)
{
  neml_assert_batch_broadcastable_dbg(a, b);
  neml_assert_dbg(a.base_dim() == 1,
                  "The first tensor in bvv has base dimension ",
                  a.base_dim(),
                  " instead of 1.");
  neml_assert_dbg(b.base_dim() == 1,
                  "The second tensor in bvv has base dimension ",
                  b.base_dim(),
                  " instead of 1.");
  return Tensor(torch::sum(a * b, -1), broadcast_batch_dim(a, b));
}
}

Tensor
operator*(const Tensor & a, const Tensor & b)
{
  neml_assert_broadcastable_dbg(a, b);
  return Tensor(torch::operator*(a, b), broadcast_batch_dim(a, b));
}
} // end namespace neml2
