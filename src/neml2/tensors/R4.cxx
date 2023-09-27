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

#include "neml2/misc/math.h"
#include "neml2/tensors/R4.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/R3.h"
#include "neml2/tensors/SSR4.h"
#include "neml2/tensors/R5.h"
#include "neml2/tensors/Rot.h"

namespace neml2
{
R4::R4(const SSR4 & T)
  : R4(math::mandel_to_full(math::mandel_to_full(T, 1)))
{
}

R4
R4::rotate(const Rot & r) const
{
  R2 R = r.euler_rodrigues();
  neml_assert_batch_broadcastable_dbg(*this, R);

  auto res = torch::einsum("...im,...jn,...ko,...lp,...mnop", {R, R, R, R, *this});
  return R4(res, broadcast_batch_dim(*this, R));
}

R5
R4::drotate(const Rot & r) const
{
  R2 R = r.euler_rodrigues();
  R3 F = r.deuler_rodrigues();
  neml_assert_batch_broadcastable_dbg(*this, R, F);

  auto res1 = torch::einsum("...jn,...ko,...lp,...mnop,...imt->...ijklt", {R, R, R, *this, F});
  auto res2 = torch::einsum("...im,...ko,...lp,...mnop,...jnt->...ijklt", {R, R, R, *this, F});
  auto res3 = torch::einsum("...im,...jn,...lp,...mnop,...kot->...ijklt", {R, R, R, *this, F});
  auto res4 = torch::einsum("...im,...jn,...ko,...mnop,...lpt->...ijklt", {R, R, R, *this, F});
  auto res = res1 + res2 + res3 + res4;

  return R5(res, broadcast_batch_dim(*this, R, F));
}

Scalar
R4::operator()(TorchSize i, TorchSize j, TorchSize k, TorchSize l) const
{
  return base_index({i, j, k, l});
}

R4
R4::transpose_minor() const
{
  return BatchTensorBase<R4>::base_transpose(0, 1).base_transpose(2, 3);
}

R4
R4::transpose_major() const
{
  return BatchTensorBase<R4>::base_transpose(0, 2).base_transpose(1, 3);
}
} // namespace neml2
