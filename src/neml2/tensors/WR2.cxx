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

#include "neml2/tensors/WR2.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/Rot.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/WSR4.h"
#include "neml2/tensors/R4.h"

#include "neml2/misc/math.h"

namespace neml2
{
WR2::WR2(const R2 & T)
  : WR2(math::full_to_skew((T - T.transpose()) / 2.0))
{
}

WR2
WR2::fill(const Real & a21,
          const Real & a02,
          const Real & a10,
          const torch::TensorOptions & options)
{
  return WR2::fill(Scalar(a21, options), Scalar(a02, options), Scalar(a10, options));
}

WR2
WR2::fill(const Scalar & a21, const Scalar & a02, const Scalar & a10)
{
  return WR2(torch::stack({a21, a02, a10}, -1), a21.batch_dim());
}

Scalar
WR2::operator()(TorchSize i, TorchSize j) const
{
  TorchSize a = math::skew_reverse_index[i][j];
  return base_index({a}) * math::skew_factor[i][j];
}

Rot
WR2::exp() const
{
  // There are singularities at norm() = 0 and pi
  // To the third order near zero this reduces to
  // *this * (1/2 + norm^2 / 24)
  // We use this formula for small rotations

  // The other singularity is essentially unavoidable

  // This is what determines which region to sit in
  auto norm = this->norm();

  // So we want the result to be as accurate as machine precision
  auto thresh = std::pow(math::eps, 1.0 / 3.0);

  // Setup with the stable Taylor series
  Rot res = (*this) * (0.5 + norm * norm / 24.0);

  // Figure out where we are relative to the cutoff
  auto gt = norm > thresh;

  // Insert the true expression
  res.index_put_({gt},
                 torch::Tensor(this->index({gt})) *
                     (torch::tan(norm.index({gt}) / 2.0) / norm.index({gt})).unsqueeze(-1));

  return res;
}

R2
WR2::dexp() const
{
  // Same singularities as WR2::exp()
  auto norm = this->norm();
  auto thresh = std::pow(math::eps, 1.0 / 3.0);
  auto gt = norm > thresh;
  auto gt_norm = norm.index({gt});

  // Stuff that goes with id, filled with the Taylor part
  auto id_factor = 0.5 + norm * norm / 24.0;
  // and backfilled with the true expression values
  id_factor.index_put_({gt}, Scalar(torch::tan(gt_norm / 2.0)) / gt_norm);

  // Stuff that goes like the outer product, filled with the Taylor part
  auto outer_factor = Scalar::zeros_like(norm) + 1.0 / 12.0;
  // and backfilled withthe true expression values
  outer_factor.index_put_(
      {gt},
      Scalar((gt_norm - torch::sin(gt_norm)) / (gt_norm * gt_norm * (1.0 + torch::cos(gt_norm)))) /
          gt_norm);

  return R2::identity(this->options()) * id_factor + this->outer(*this) * outer_factor;
}

} // namespace neml2
