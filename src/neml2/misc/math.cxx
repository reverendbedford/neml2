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
#include "neml2/misc/error.h"
#include "neml2/tensors/tensors.h"

namespace neml2
{
namespace math
{
const torch::Tensor
full_to_mandel_map(const torch::TensorOptions & options)
{
  return torch::tensor({0, 4, 8, 5, 2, 1}, options);
}
const torch::Tensor
mandel_to_full_map(const torch::TensorOptions & options)
{
  return torch::tensor({0, 5, 4, 5, 1, 3, 4, 3, 2}, options);
}
const torch::Tensor
full_to_mandel_factor(const torch::TensorOptions & options)
{
  return torch::tensor({1.0, 1.0, 1.0, sqrt2, sqrt2, sqrt2}, options);
}
const torch::Tensor
mandel_to_full_factor(const torch::TensorOptions & options)
{
  return torch::tensor({1.0, invsqrt2, invsqrt2, invsqrt2, 1.0, invsqrt2, invsqrt2, invsqrt2, 1.0},
                       options);
}

const torch::Tensor
full_to_skew_map(const torch::TensorOptions & options)
{
  return torch::tensor({7, 2, 3}, options);
}
const torch::Tensor
skew_to_full_map(const torch::TensorOptions & options)
{
  return torch::tensor({0, 2, 1, 2, 0, 0, 1, 0, 0}, options);
}
const torch::Tensor
full_to_skew_factor(const torch::TensorOptions & options)
{
  return torch::tensor({1.0, 1.0, 1.0}, options);
}
const torch::Tensor
skew_to_full_factor(const torch::TensorOptions & options)
{
  return torch::tensor({0.0, -1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 1.0, 0.0}, options);
}

BatchTensor
full_to_reduced(const BatchTensor & full,
                const torch::Tensor & rmap,
                const torch::Tensor & rfactors,
                TorchSize dim)
{
  using namespace torch::indexing;

  auto batch_dim = full.batch_dim();
  auto starting_dim = batch_dim + dim;
  auto trailing_dim = full.dim() - starting_dim - 2; // 2 comes from the reduced axes (3,3)
  auto starting_shape = full.sizes().slice(0, starting_dim);
  auto trailing_shape = full.sizes().slice(starting_dim + 2);

  TorchSlice net(starting_dim, None);
  net.push_back(Ellipsis);
  net.insert(net.end(), trailing_dim, None);
  auto map =
      rmap.index(net).expand(utils::add_shapes(starting_shape, rmap.sizes()[0], trailing_shape));
  auto factor = rfactors.to(full).index(net);

  return BatchTensor(
      factor * torch::gather(full.reshape(utils::add_shapes(starting_shape, 9, trailing_shape)),
                             starting_dim,
                             map),
      batch_dim);
}

BatchTensor
reduced_to_full(const BatchTensor & reduced,
                const torch::Tensor & rmap,
                const torch::Tensor & rfactors,
                TorchSize dim)
{
  using namespace torch::indexing;

  auto batch_dim = reduced.batch_dim();
  auto starting_dim = batch_dim + dim;
  auto trailing_dim = reduced.dim() - starting_dim - 1; // There's only 1 axis to unsqueeze
  auto starting_shape = reduced.sizes().slice(0, starting_dim);
  auto trailing_shape = reduced.sizes().slice(starting_dim + 1);

  TorchSlice net(starting_dim, None);
  net.push_back(Ellipsis);
  net.insert(net.end(), trailing_dim, None);
  auto map = rmap.index(net).expand(utils::add_shapes(starting_shape, 9, trailing_shape));
  auto factor = rfactors.to(reduced).index(net);

  return BatchTensor((factor * torch::gather(reduced, starting_dim, map))
                         .reshape(utils::add_shapes(starting_shape, 3, 3, trailing_shape)),
                     batch_dim);
}

BatchTensor
full_to_mandel(const BatchTensor & full, TorchSize dim)
{
  return full_to_reduced(full,
                         full_to_mandel_map(full.options().dtype(TORCH_INT_DTYPE)),
                         full_to_mandel_factor(full.options()),
                         dim);
}

BatchTensor
mandel_to_full(const BatchTensor & mandel, TorchSize dim)
{
  return reduced_to_full(mandel,
                         mandel_to_full_map(mandel.options().dtype(TORCH_INT_DTYPE)),
                         mandel_to_full_factor(mandel.options()),
                         dim);
}

BatchTensor
full_to_skew(const BatchTensor & full, TorchSize dim)
{
  return full_to_reduced(full,
                         full_to_skew_map(full.options().dtype(TORCH_INT_DTYPE)),
                         full_to_skew_factor(full.options()),
                         dim);
}

BatchTensor
skew_to_full(const BatchTensor & skew, TorchSize dim)
{
  return reduced_to_full(skew,
                         skew_to_full_map(skew.options().dtype(TORCH_INT_DTYPE)),
                         skew_to_full_factor(skew.options()),
                         dim);
}

BatchTensor
jacrev(const BatchTensor & out, const BatchTensor & p)
{
  neml_assert_dbg(
      p.batch_dim() == 0 || out.batch_sizes() == p.batch_sizes(),
      "If the parameter is batched, its batch shape must be the same as the batch shape "
      "of the output. However, the batch shape of the parameter is ",
      p.batch_sizes(),
      ", and the batch shape of the output is ",
      out.batch_sizes());

  // flatten out to handle arbitrarily shaped output
  auto outf = BatchTensor(
      out.reshape(utils::add_shapes(out.batch_sizes(), utils::storage_size(out.base_sizes()))),
      out.batch_dim());

  neml_assert_dbg(outf.base_dim() == 1, "Flattened output must be flat.");

  auto doutf_dp = BatchTensor::empty(
      outf.batch_sizes(), utils::add_shapes(outf.base_sizes(), p.base_sizes()), outf.options());

  for (TorchSize i = 0; i < outf.base_sizes()[0]; i++)
  {
    auto G = BatchTensor::zeros_like(outf);
    G.index_put_({torch::indexing::Ellipsis, i}, 1.0);
    auto doutfi_dp = torch::autograd::grad({outf},
                                           {p},
                                           {G},
                                           /*retain_graph=*/true,
                                           /*create_graph=*/false,
                                           /*allow_unused=*/false)[0];
    if (doutfi_dp.defined())
      doutf_dp.base_index_put({i, torch::indexing::Ellipsis}, doutfi_dp);
  }

  // reshape the derivative back to the correct shape
  auto dout_dp = BatchTensor(
      doutf_dp.reshape(utils::add_shapes(out.batch_sizes(), out.base_sizes(), p.base_sizes())),
      out.batch_dim());

  // factor to account for broadcasting
  Real factor = p.batch_dim() == 0 ? utils::storage_size(out.batch_sizes()) : 1;

  return dout_dp / factor;
}

BatchTensor
base_diag_embed(const BatchTensor & a, TorchSize offset, TorchSize d1, TorchSize d2)
{
  return BatchTensor(
      torch::diag_embed(
          a, offset, d1 < 0 ? d1 : d1 + a.batch_dim() + 1, d2 < 0 ? d2 : d2 + a.batch_dim() + 1),
      a.batch_dim());
}

SR2
skew_and_sym_to_sym(const SR2 & e, const WR2 & w)
{
  // In NEML we used an unrolled form, I don't think I ever found
  // a nice direct notation for this one
  auto E = R2(e);
  auto W = R2(w);
  return SR2(W * E - E * W);
}

SSR4
d_skew_and_sym_to_sym_d_sym(const WR2 & w)
{
  auto I = R2::identity(w.options());
  auto W = R2(w);
  return SSR4(R4(torch::einsum("...ia,...jb->...ijab", {W, I}) -
                 torch::einsum("...ia,...bj->...ijab", {I, W})));
}

SWR4
d_skew_and_sym_to_sym_d_skew(const SR2 & e)
{
  auto I = R2::identity(e.options());
  auto E = R2(e);
  return SWR4(R4(torch::einsum("...ia,...bj->...ijab", {I, E}) -
                 torch::einsum("...ia,...jb->...ijab", {E, I})));
}

WR2
multiply_and_make_skew(const SR2 & a, const SR2 & b)
{
  auto A = R2(a);
  auto B = R2(b);

  return WR2(A * B - B * A);
}

WSR4
d_multiply_and_make_skew_d_first(const SR2 & b)
{
  auto I = R2::identity(b.options());
  auto B = R2(b);
  return WSR4(R4(torch::einsum("...ia,...bj->...ijab", {I, B}) -
                 torch::einsum("...ia,...jb->...ijab", {B, I})));
}

WSR4
d_multiply_and_make_skew_d_second(const SR2 & a)
{
  auto I = R2::identity(a.options());
  auto A = R2(a);
  return WSR4(R4(torch::einsum("...ia,...jb->...ijab", {A, I}) -
                 torch::einsum("...ia,...bj->...ijab", {I, A})));
}

namespace linalg
{
std::tuple<BatchTensor, BatchTensor>
lu_factor(const BatchTensor & A, bool pivot)
{
  auto [LU, pivots] = torch::linalg_lu_factor(A, pivot);
  return {BatchTensor(LU, A.batch_dim()), BatchTensor(pivots, A.batch_dim())};
}

BatchTensor
lu_solve(const BatchTensor & LU,
         const BatchTensor & pivots,
         const BatchTensor & B,
         bool left,
         bool adjoint)
{
  return BatchTensor(torch::linalg_lu_solve(LU, pivots, B, left, adjoint), B.batch_dim());
}
} // namespace linalg
} // namespace math
} // namespace neml2
